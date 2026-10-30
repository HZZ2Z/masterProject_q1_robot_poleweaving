"""Bounded motor-pretraining loop. Current reference: neutral standing only.

No soccer acceptance is inferred from this stage. Run with --help for bounds.
"""
import argparse
import copy
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from q1_supermimic.physics import WarpPhysics
from q1_supermimic.robot import Q1Control
from q1_supermimic.observations import MotorObservation
from q1_supermimic.tracking import TrackingHead
from q1_supermimic.dynamics import Dynamics
from q1_supermimic.networks import MotorPolicy, mlp
from q1_supermimic.replay import Replay
from q1_supermimic.learner import tracking_update


def reference(model, control, device, count):
    state = torch.tensor(np.r_[model.qpos0, np.zeros(model.nv)], device=device, dtype=torch.float32).repeat(count, 1)
    state[:, control.qpos_indices] = control.neutral
    return state


@torch.no_grad()
def collect(physics, observe, policy, replay, steps, generator):
    model, device = physics.native_model, physics.device
    count = physics.qpos.shape[0]
    ref = reference(model, observe.control, device, count)
    physics.reset(ref[:, :model.nq], ref[:, model.nq:])
    previous = torch.zeros(count, 22, device=device)
    history = observe(ref, previous).unsqueeze(1).repeat(1, 10, 1)
    goal = observe.reference(ref).unsqueeze(1).repeat(1,10,1).flatten(1)
    alive = torch.ones(count,dtype=torch.bool,device=device)
    for t in range(steps):
        state = torch.cat((physics.qpos, physics.qvel), -1).clone()
        action = policy.track(history.flatten(1), goal)
        action = action + .05 * torch.randn(action.shape, generator=generator, device=device)
        for _ in range(observe.control.substeps):
            physics.step(observe.control(physics.qpos, physics.qvel, action))
        following = torch.cat((physics.qpos, physics.qvel), -1).clone()
        # Same hard termination as the imagined tracking head.
        root = following[:, 3:7]
        gravity_z = 2 * (root[:,1].square() + root[:,2].square()) - 1
        fell = ((following[:,2] - ref[:,2]).abs() > .25) | ((gravity_z + 1).abs() > .8)
        replay.append(state=state, next_state=following, action=action,
                      boundary=fell | (t == steps-1), terminated=fell,
                      context=torch.zeros(count,0,device=device), history=history, valid=alive)
        alive = alive & ~fell
        sensor = observe(following, action)
        history = torch.cat((history[:,1:],sensor.unsqueeze(1)),1)
    physics.check()


@torch.no_grad()
def evaluate(model, policy, observe, seed=9187, episodes=4, details=False):
    actor, sensor = copy.deepcopy(policy).cpu(), copy.deepcopy(observe).cpu()
    control = sensor.control
    ref = reference(model, control, 'cpu', 1)
    goal = sensor.reference(ref).unsqueeze(1).repeat(1,10,1).flatten(1)
    random = np.random.default_rng(seed)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    falls, errors, returns, episode_falls = 0, [], [], []
    head = TrackingHead(model)
    for _ in range(episodes):
        fallen = False
        recorded_states, recorded_actions = [], []
        data = mujoco.MjData(model)
        data.qpos[:] = ref[0,:model.nq].numpy()
        data.qpos[control.qpos_indices.numpy()] += random.normal(0,.005,22)
        history = sensor(torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32).unsqueeze(0),torch.zeros(1,22)).unsqueeze(1).repeat(1,10,1)
        for _ in range(250):
            action = actor.track(history.flatten(1),goal)
            for _ in range(control.substeps):
                state = torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32).unsqueeze(0)
                data.ctrl[:] = control(state[:,:model.nq],state[:,model.nq:],action)[0].numpy()
                mujoco.mj_step(model,data)
            state = torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32).unsqueeze(0)
            recorded_states.append(state[0].clone())
            recorded_actions.append(action[0].clone())
            history = torch.cat((history[:,1:],sensor(state,action).unsqueeze(1)),1)
            errors.append(float(np.mean((data.qpos[control.qpos_indices.numpy()] - control.neutral.numpy())**2)))
            q = data.qpos[3:7]
            if abs(data.qpos[2]-ref[0,2].item())>.25 or 2*(q[1]**2+q[2]**2)>.8:
                falls += 1
                fallen = True
                break
        states = torch.stack(recorded_states)
        actions = torch.stack(recorded_actions)
        previous = torch.cat((torch.zeros_like(actions[:1]),actions[:-1]),0)
        rewards, _, _ = head(states,ref.expand_as(states),actions,previous)
        # Fixed full-horizon denominator: early falls must reduce return.
        returns.append(rewards.sum().item() / 25.)
        episode_falls.append(fallen)
    result = dict(episodes=episodes, seconds_per_episode=5, fall_rate=falls/episodes,
                  tracking_return=float(np.mean(returns)), joint_mse=float(np.mean(errors)))
    if details:
        result.update(episode_returns=returns, episode_falls=episode_falls)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=5, help='Additional bounded iterations')
    parser.add_argument('--output', type=Path, default=Path('logs/motor_pretraining'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(41)
    torch.cuda.set_per_process_memory_fraction(.4)
    physics = WarpPhysics(16)
    model, device = physics.native_model, physics.device
    observe, head = MotorObservation(model).to(device), TrackingHead(model).to(device)
    world = Dynamics(model,widths=(256,256,256)).to(device)
    policy = MotorPolicy(720,540,4,22,motor_widths=(128,128),visual_widths=(64,64)).to(device)
    # Start from the independently validated neutral PD controller.
    torch.nn.init.zeros_(policy.decoder[-1].weight)
    torch.nn.init.zeros_(policy.decoder[-1].bias)
    critic = mlp(610,1,(256,128)).to(device)
    wm_opt = torch.optim.Adam(world.parameters(),lr=2e-4)
    actor_opt = torch.optim.Adam(policy.parameters(),lr=2e-4,betas=(.9,.95))
    critic_opt = torch.optim.Adam(critic.parameters(),lr=2e-4)
    replay = Replay(256)
    generator = torch.Generator(device=device).manual_seed(731)
    start = 0
    if args.resume:
        saved = torch.load(args.resume,map_location=device,weights_only=False)
        for name, module in [('world',world),('policy',policy),('critic',critic)]:
            module.load_state_dict(saved[name])
        for name, optimizer in [('wm_opt',wm_opt),('actor_opt',actor_opt),('critic_opt',critic_opt)]:
            optimizer.load_state_dict(saved[name])
        replay.storage, replay.size, replay.cursor = saved['replay'],saved['replay_size'],saved['replay_cursor']
        generator.set_state(saved['generator'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        start = saved['iteration']
    else:
        collect(physics,observe,policy,replay,160,generator)
        world.set_scales(replay.storage['state'][:replay.size])
        for _ in range(200):
            wm_opt.zero_grad(set_to_none=True)
            loss = world.window_loss(replay.sample(128))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True)
            wm_opt.step()
    args.output.mkdir(parents=True,exist_ok=True)
    initial = evaluate(model,policy,observe)
    print(json.dumps(dict(iteration=start,evaluation=initial)),flush=True)
    for iteration in range(start+1,start+args.iterations+1):
        collect(physics,observe,policy,replay,64,generator)
        for _ in range(10):
            wm_opt.zero_grad(set_to_none=True)
            loss = world.window_loss(replay.sample(128))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True)
            wm_opt.step()
        batch = replay.sample(8,length=1)
        ref = reference(model,observe.control,device,8).unsqueeze(0).repeat(43,1,1)
        result = tracking_update(policy,world,critic,head,observe,actor_opt,critic_opt,
                                 batch['state'][0],batch['history'][0],ref,batch['context'][0])
        result.update(iteration=iteration,model_loss=loss.item())
        if iteration == start+args.iterations:
            result['evaluation'] = evaluate(model,policy,observe)
        print(json.dumps(result),flush=True)
        with (args.output/'metrics.jsonl').open('a') as log:
            log.write(json.dumps(result)+'\n')
        saved = dict(iteration=iteration,world=world.state_dict(),policy=policy.state_dict(),critic=critic.state_dict(),
                     wm_opt=wm_opt.state_dict(),actor_opt=actor_opt.state_dict(),critic_opt=critic_opt.state_dict(),
                     replay=replay.storage,replay_size=replay.size,replay_cursor=replay.cursor,
                     generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
                     scope='neutral standing motor pretraining; not soccer',
                     evaluation=result.get('evaluation'), deployment_accepted=False)
        temporary = args.output/'checkpoint.tmp'
        torch.save(saved,temporary)
        temporary.replace(args.output/'last.pt')


if __name__ == '__main__':
    main()
