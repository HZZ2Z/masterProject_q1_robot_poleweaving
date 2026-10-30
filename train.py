"""Bounded motor-pretraining loop. Current reference: neutral standing only.

No soccer acceptance is inferred from this stage. Run with --help for bounds.
"""
import argparse
import copy
import json
import os
import signal
import time
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
def collect(physics, observe, policy, replay, steps, generator, noise=.05, perturbation=0.):
    model, device = physics.native_model, physics.device
    count = physics.qpos.shape[0]
    ref = reference(model, observe.control, device, count)
    initial = ref.clone()
    initial[:, observe.control.qpos_indices] += perturbation * torch.randn(
        (count, 22), generator=generator, device=device)
    physics.reset(initial[:, :model.nq], initial[:, model.nq:])
    previous = torch.zeros(count, 22, device=device)
    history = observe(initial, previous).unsqueeze(1).repeat(1, 10, 1)
    goal = observe.reference(ref).unsqueeze(1).repeat(1,10,1).flatten(1)
    alive = torch.ones(count,dtype=torch.bool,device=device)
    for t in range(steps):
        state = torch.cat((physics.qpos, physics.qvel), -1).clone()
        action = policy.track(history.flatten(1), goal)
        action = action + noise * torch.randn(action.shape, generator=generator, device=device)
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=5, help='Additional iterations')
    parser.add_argument('--output', type=Path, default=Path('logs/motor_pretraining'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--model-batch-size', type=int, default=128)
    parser.add_argument('--policy-batch-size', type=int, default=8)
    parser.add_argument('--model-width', type=int, default=256)
    parser.add_argument('--policy-width', type=int, default=128)
    parser.add_argument('--warmup-batches', type=int, default=200)
    parser.add_argument('--model-batches', type=int, default=10)
    parser.add_argument('--collect-steps', type=int, default=64)
    parser.add_argument('--replay-steps', type=int, default=256)
    parser.add_argument('--exploration-std', type=float, default=.01)
    parser.add_argument('--initial-perturbation', type=float, default=.005)
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--evaluation-interval', type=int, default=5)
    parser.add_argument('--evaluation-episodes', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--memory-fraction', type=float, default=.4)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-entity', default='mohismlab-robotics')
    parser.add_argument('--wandb-project', default='q1-humanoid-kick-SuperMimic')
    parser.add_argument('--run-name')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--max-rejected-evaluations', type=int, default=20)
    args = parser.parse_args()
    counts = ('iterations','num_envs','model_batch_size','policy_batch_size','model_width',
              'policy_width','warmup_batches','model_batches','collect_steps','replay_steps',
              'evaluation_interval','evaluation_episodes','save_interval','threads',
              'max_rejected_evaluations')
    if any(getattr(args, key) < 1 for key in counts):
        parser.error('Counts and sizes must be positive')
    if args.collect_steps < 8 or args.replay_steps < max(160, args.collect_steps):
        parser.error('Need at least 8 collection steps and replay capacity for warmup (160) and collection')
    if not 0 < args.memory_fraction <= .95 or args.exploration_std < 0 or args.initial_perturbation < 0 or args.actor_lr <= 0:
        parser.error('Invalid memory fraction, noise, perturbation or learning rate')
    return args


def retain_candidate(candidate, baseline, pending):
    """Selection on paired real episodes; separate audit still required."""
    values = [candidate['fall_rate'], candidate['tracking_return'],
              baseline['fall_rate'], baseline['tracking_return']]
    return bool(pending and all(np.isfinite(values)) and
                candidate['fall_rate'] <= baseline['fall_rate'] and
                candidate['tracking_return'] >= baseline['tracking_return'])


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config.update(scope='neutral standing motor pretraining; not soccer',
                  deployment_accepted=False, gpu=torch.cuda.get_device_name(0),
                  vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                  torch_version=torch.__version__, cpu_threads=os.cpu_count())
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_float32_matmul_precision('high')
    physics = WarpPhysics(args.num_envs)
    model, device = physics.native_model, physics.device
    observe, head = MotorObservation(model).to(device), TrackingHead(model).to(device)
    world = Dynamics(model,widths=(args.model_width,)*3).to(device)
    policy = MotorPolicy(720,540,4,22,motor_widths=(args.policy_width,)*2,visual_widths=(64,64)).to(device)
    torch.nn.init.zeros_(policy.decoder[-1].weight)
    torch.nn.init.zeros_(policy.decoder[-1].bias)
    critic = mlp(610,1,(args.model_width,args.model_width//2)).to(device)
    wm_opt = torch.optim.Adam(world.parameters(),lr=2e-4)
    actor_opt = torch.optim.Adam(policy.parameters(),lr=args.actor_lr,betas=(.9,.95))
    critic_opt = torch.optim.Adam(critic.parameters(),lr=2e-4)
    replay = Replay(args.replay_steps)
    generator = torch.Generator(device=device).manual_seed(731)
    start = 0
    saved = None
    if args.resume:
        saved = torch.load(args.resume,map_location=device,weights_only=False)
        if saved.get('config'):
            for key in ('num_envs','model_width','policy_width','replay_steps'):
                if saved['config'][key] != config[key]:
                    raise ValueError(f'Resume must preserve {key}')
        for name,module in [('world',world),('policy',policy),('critic',critic)]:
            module.load_state_dict(saved[name])
        for name,optimizer in [('wm_opt',wm_opt),('actor_opt',actor_opt),('critic_opt',critic_opt)]:
            optimizer.load_state_dict(saved[name])
        replay.storage,replay.size,replay.cursor = saved['replay'],saved['replay_size'],saved['replay_cursor']
        if replay.storage['state'].shape[1] != args.num_envs or replay.storage['state'].shape[0] != args.replay_steps:
            raise ValueError('Resume replay shape mismatch')
        generator.set_state(saved['generator'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        start = saved['iteration']
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity,project=args.wandb_project,
                         name=args.run_name,config=config,dir=str(args.output),mode='online',
                         id=saved.get('wandb_run_id') if saved else None,
                         resume='must' if saved and saved.get('wandb_run_id') else None)
        config['wandb_url'] = run.url
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        print(json.dumps({'wandb_url':run.url}),flush=True)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)

    def log(values, step):
        print(json.dumps(values),flush=True)
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(values)+'\n')
        if run:
            def flatten(mapping, prefix=''):
                result={}
                for key,value in mapping.items():
                    name=prefix+key
                    if isinstance(value,dict):
                        result.update(flatten(value,name+'/'))
                    elif isinstance(value,(int,float,bool)):
                        result[name]=value
                return result
            run.log(flatten(values),step=step)

    collection_policy = copy.deepcopy(policy)
    accepted_optimizer = copy.deepcopy(actor_opt.state_dict())
    rejected = saved.get('rejected_evaluations',0) if saved else 0
    if saved and saved.get('accepted_policy'):
        collection_policy.load_state_dict(saved['accepted_policy'])
        accepted_optimizer=saved['accepted_actor_optimizer']
    # An independent collection seed provides held-out complete trajectories.
    validation = Replay(160)
    validation_generator = torch.Generator(device=device).manual_seed(194917)
    collect(physics,observe,collection_policy,validation,160,validation_generator,
            args.exploration_std,args.initial_perturbation)
    held_out=validation.sample(args.model_batch_size,generator=validation_generator)
    del validation
    if not saved:
        collect(physics,observe,policy,replay,160,generator,args.exploration_std,args.initial_perturbation)
        world.set_scales(replay.storage['state'][:replay.size])
        for batch_index in range(args.warmup_batches):
            wm_opt.zero_grad(set_to_none=True)
            loss = world.window_loss(replay.sample(args.model_batch_size))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True)
            wm_opt.step()
            if (batch_index+1)%100==0:
                print(json.dumps({'warmup_batch':batch_index+1,'model_loss':loss.item()}),flush=True)
    baseline = evaluate(model,collection_policy,observe,episodes=args.evaluation_episodes)
    log(dict(iteration=start,evaluation=baseline),start)
    pending=saved.get('pending_actor_updates',0) if saved else 0
    last_retained=saved.get('last_retained_iteration',start) if saved else start
    iteration=start
    started=time.perf_counter()
    reason='iterations_complete'

    def checkpoint():
        payload=dict(iteration=iteration,world=world.state_dict(),policy=policy.state_dict(),critic=critic.state_dict(),
                     wm_opt=wm_opt.state_dict(),actor_opt=actor_opt.state_dict(),critic_opt=critic_opt.state_dict(),
                     accepted_policy=collection_policy.state_dict(),accepted_actor_optimizer=accepted_optimizer,
                     replay=replay.storage,replay_size=replay.size,replay_cursor=replay.cursor,
                     generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
                     config=config,wandb_run_id=run.id if run else None,rejected_evaluations=rejected,
                     pending_actor_updates=pending,last_retained_iteration=last_retained,
                     scope=config['scope'],evaluation=baseline,deployment_accepted=False)
        temporary=args.output/'checkpoint.tmp'
        torch.save(payload,temporary)
        temporary.replace(args.output/'last.pt')

    # Persist warmup before the first large actor allocation.
    checkpoint()
    for iteration in range(start+1,start+args.iterations+1):
        tick=time.perf_counter()
        collect(physics,observe,collection_policy,replay,args.collect_steps,generator,
                args.exploration_std,args.initial_perturbation)
        torch.cuda.synchronize()
        collection_seconds=time.perf_counter()-tick
        for _ in range(args.model_batches):
            wm_opt.zero_grad(set_to_none=True)
            loss=world.window_loss(replay.sample(args.model_batch_size))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True)
            wm_opt.step()
        with torch.no_grad():
            validation_loss=world.window_loss(held_out).item()
            persistence_loss=world.error(held_out['state'][0].expand_as(held_out['next_state']),held_out['next_state']).item()
        # Prediction error is a prerequisite, never evidence of useful control.
        model_ready=np.isfinite(validation_loss) and validation_loss < persistence_loss
        result=dict(iteration=iteration,model_loss=loss.item(),validation=dict(
                    model_loss=validation_loss,persistence_loss=persistence_loss,beats_persistence=model_ready))
        if model_ready:
            batch=replay.sample(args.policy_batch_size,length=1)
            ref=reference(model,observe.control,device,args.policy_batch_size).unsqueeze(0).repeat(43,1,1)
            result.update(tracking_update(policy,world,critic,head,observe,actor_opt,critic_opt,
                         batch['state'][0],batch['history'][0],ref,batch['context'][0]))
            pending+=1
        result['actor_update_applied']=bool(model_ready)
        if iteration%args.evaluation_interval==0 or iteration==start+args.iterations or stop:
            candidate=evaluate(model,policy,observe,episodes=args.evaluation_episodes)
            # Same initial conditions as incumbent. This is a selection check,
            # NOT deployment acceptance; audit.py uses a separate seed/cohort.
            retained=retain_candidate(candidate,baseline,pending)
            result['evaluation']=candidate
            result['selection']=dict(retained=retained,return_difference=candidate['tracking_return']-baseline['tracking_return'],
                                     pending_actor_updates=pending)
            if retained:
                collection_policy.load_state_dict(policy.state_dict())
                accepted_optimizer=copy.deepcopy(actor_opt.state_dict())
                baseline=candidate
                last_retained=iteration
                rejected=0
            elif pending:
                policy.load_state_dict(collection_policy.state_dict())
                actor_opt.load_state_dict(accepted_optimizer)
                rejected+=1
            pending=0
            result['selection']['consecutive_rejected_evaluations']=rejected
        torch.cuda.synchronize()
        result['performance']=dict(iteration_seconds=time.perf_counter()-tick,
              collection_transitions_per_second=args.num_envs*args.collect_steps/collection_seconds,
              gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
              gpu_reserved_mib=torch.cuda.memory_reserved()/2**20,
              gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        stalled=iteration-last_retained >= args.max_rejected_evaluations*args.evaluation_interval
        result['iterations_since_retained_update']=iteration-last_retained
        log(result,iteration)
        if iteration%args.save_interval==0 or iteration==start+args.iterations or stop or stalled:
            checkpoint()
        if stop:
            reason='signal';break
        if stalled:
            reason='no_retained_policy_progress';break
    status=dict(pid=os.getpid(),iteration=iteration,reason=reason,elapsed_seconds=time.perf_counter()-started,
                incumbent_evaluation=baseline,wandb_url=run.url if run else None,deployment_accepted=False)
    (args.output/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    if run:
        run.summary.update(status)
        run.finish()


if __name__ == '__main__':
    main()
