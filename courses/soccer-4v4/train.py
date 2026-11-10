"""Explicit CLI training only; no jobs are started by setup or preview."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import signal
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from q1_4v4.env import SoccerEnv
from q1_4v4.learning import make_args, make_team, numpy, save_team, load_team, policy_sharing, with_team_identity, configure_temporal
from q1_4v4.curriculum import Curriculum, STAGES
from q1_4v4.course_config import COURSES


def run(options):
    if options.backend == 'warp':
        from q1_4v4.gpu_runner import run as run_gpu
        return run_gpu(options)
    if getattr(options,'curriculum_mode',None)=='parallel':raise ValueError('Parallel curricula require the Warp backend')
    if getattr(options,'match_motion_constraints',False):raise ValueError('Match motion constraints require the Warp backend')
    if options.episode_steps is not None and options.episode_steps < 1:
        raise ValueError('episode-steps must be positive')
    if options.save_interval < 1:
        raise ValueError('save-interval must be positive')
    if min(options.iterations, options.num_envs, options.horizon, options.epochs) < 1:
        raise ValueError('Iterations, envs, horizon and epochs must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    device = torch.device(options.device)
    args = make_args(options.num_envs, options.horizon, options.epochs)
    curriculum = Curriculum(options.stage)
    envs = [SoccerEnv(options.seed, stage=curriculum.stage)]
    envs += [SoccerEnv(options.seed+i, stage=curriculum.stage, model=envs[0].model) for i in range(1, options.num_envs)]
    checkpoint = torch.load(options.checkpoint,map_location=device,weights_only=False) if options.checkpoint else None
    configure_temporal(args,options,checkpoint)
    shared = policy_sharing(options,checkpoint)
    groups = [slice(None)] if shared else [slice(0,4),slice(4,8)]
    agents = 8 if shared else 4
    teams = [make_team(args, envs[0], device,shared=shared) for _ in groups]
    def inputs(obs,state):
        if shared:
            return with_team_identity(obs,np.arange(8)//4),with_team_identity(state,np.arange(8)//4,role_suffix=4)
        return obs,state
    start = 0
    if options.checkpoint:
        if len(checkpoint['teams'])!=len(teams):raise ValueError('Checkpoint policy count mismatch')
        if checkpoint['args']['hidden_size'] != args.hidden_size:
            raise ValueError('Checkpoint network configuration mismatch')
        for state, (policy, trainer, _) in zip(checkpoint['teams'], teams):
            load_team(state, policy, trainer)
        curriculum.load_state_dict(checkpoint['curriculum'])
        start = checkpoint['iteration']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(options), indent=2))
    writer = SummaryWriter(str(out/'tensorboard'))
    wandb_run = None
    if options.wandb:
        import wandb
        wandb_run = wandb.init(entity='mohismlab-robotics', project='q1-humanoid-kick-4v4',
                              name=options.run_name or out.name, job_type=options.job_type,
                              tags=['q1', '4v4', 'mappo', options.job_type],
                              config=vars(options), dir=str(out.resolve()),
                              settings=wandb.Settings(init_timeout=60))
        wandb_run.define_metric('iteration')
        wandb_run.define_metric('*', step_metric='iteration')
        (out/'wandb_run.json').write_text(json.dumps({'id': wandb_run.id, 'url': wandb_run.url}, indent=2))
    pool = ThreadPoolExecutor(max_workers=options.num_envs)
    stop = False

    def stop_handler(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def seed_buffers():
        duration = dict(balance=300, move=500, approach=500, kick=500, teamplay=1000, match=1500)
        for env in envs:
            env.stage = curriculum.stage
            env.episode_steps = options.episode_steps or duration[env.stage]
            env.reset()
        for t, (_, _, b) in enumerate(teams):
            b.step = 0
            b.obs[0] = np.stack([inputs(e.observe(),e.state())[0][groups[t]] for e in envs])
            b.share_obs[0] = np.stack([inputs(e.observe(),e.state())[1][groups[t]] for e in envs])
            b.rnn_states[0] = 0
            b.rnn_states_critic[0] = 0
            b.masks[0] = 1

    def save(iteration, name='last.pt'):
        payload = dict(version=1, policy_sharing=options.policy_sharing, iteration=iteration, args=vars(args),
                       teams=[save_team(p, t) for p, t, _ in teams],
                       curriculum=curriculum.state_dict(), torch_rng=torch.get_rng_state(),
                       numpy_rng=np.random.get_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temp = out/(name+'.tmp')
        torch.save(payload, temp)
        temp.replace(out/name)

    seed_buffers()
    iteration = start
    try:
        for iteration in range(start+1, start+options.iterations+1):
            tic = time.perf_counter()
            episode_infos = []
            for step in range(options.horizon):
                samples = []
                with torch.no_grad():
                    for policy, trainer, b in teams:
                        trainer.prep_rollout()
                        flatten = lambda x: x.reshape(-1, *x.shape[2:])
                        sample = policy.get_actions(flatten(b.share_obs[step]), flatten(b.obs[step]),
                                                    flatten(b.rnn_states[step]), flatten(b.rnn_states_critic[step]),
                                                    flatten(b.masks[step]))
                        samples.append([numpy(x).reshape(options.num_envs, agents, *x.shape[1:]) for x in sample])
                actions = np.concatenate([s[1] for s in samples], axis=1)
                results = list(pool.map(lambda pair: pair[0].step(pair[1]), zip(envs, actions)))
                for t, (policy, trainer, b) in enumerate(teams):
                    sl = groups[t]
                    values, acts, logs, ra, rc = samples[t]
                    obs = np.stack([inputs(r[0],r[1])[0][sl] for r in results])
                    state = np.stack([inputs(r[0],r[1])[1][sl] for r in results])
                    rewards = np.stack([r[2][sl] for r in results])
                    masks = np.ones((options.num_envs, agents, 1), dtype=np.float32)
                    for n, result in enumerate(results):
                        if result[4]:
                            # Bootstrap final observation BEFORE resetting a timeout.
                            with torch.no_grad():
                                v = policy.get_values(state[n], rc[n], np.ones((agents, 1), np.float32))
                                rewards[n] += args.gamma*numpy(trainer.value_normalizer.denormalize(v))
                        if result[3] or result[4]:
                            masks[n] = 0
                            ra[n] = rc[n] = 0
                    # One reset per world, shared by both teams, after both terminal values are read.
                    samples[t] = [state, obs, ra, rc, acts, logs, values, rewards, masks]
                for n, result in enumerate(results):
                    if result[3] or result[4]:
                        episode_infos.append(result[5])
                        curriculum.record(result[5])
                        obs, state = inputs(*envs[n].reset())
                        for t,sl in enumerate(groups):
                            samples[t][0][n] = state[sl]
                            samples[t][1][n] = obs[sl]
                for (_, _, b), sample in zip(teams, samples):
                    b.insert(*sample)
            metrics = {}
            for t, (policy, trainer, b) in enumerate(teams):
                with torch.no_grad():
                    value = policy.get_values(b.share_obs[-1].reshape(-1, b.share_obs.shape[-1]),
                                              b.rnn_states_critic[-1].reshape(-1, args.recurrent_N, args.hidden_size),
                                              b.masks[-1].reshape(-1, 1))
                b.compute_returns(numpy(value).reshape(options.num_envs, agents, 1), trainer.value_normalizer)
                trainer.prep_training()
                info = trainer.train(b)
                for k, v in info.items():
                    metrics[f'{"shared" if shared else ("red", "blue")[t]}/{k}'] = float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                if not all(torch.isfinite(p).all() for p in policy.actor.parameters()):
                    raise FloatingPointError('Nonfinite actor parameters')
                if shared:
                    metrics['red/reward']=float(b.rewards[:,:,:4].mean())
                    metrics['blue/reward']=float(b.rewards[:,:,4:].mean())
                else:metrics[f'{("red", "blue")[t]}/reward'] = float(b.rewards.mean())
                b.after_update()
            if not all(np.isfinite(x) for x in metrics.values()):
                raise FloatingPointError('Nonfinite MAPPO update')
            metrics.update(iteration=iteration, seconds=time.perf_counter()-tic,
                           episodes=len(episode_infos), goals=sum(bool(i['goal']) for i in episode_infos),
                           falls=sum(any(i['fallen']) for i in episode_infos), curriculum=curriculum.metrics())
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics)+'\n')
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    writer.add_scalar(k, v, iteration)
            for k, v in curriculum.metrics().items():
                if isinstance(v, (float, int)):
                    writer.add_scalar('curriculum/'+k, v, iteration)
            if wandb_run:
                cloud = {k: v for k, v in metrics.items() if k != 'curriculum'}
                cloud.update({'curriculum/'+k: v for k, v in curriculum.metrics().items()})
                cloud['curriculum/stage_index'] = curriculum.index
                wandb_run.log(cloud, step=iteration)
            writer.flush()
            print(json.dumps(metrics), flush=True)
            old_stage = curriculum.stage
            old_curriculum = curriculum.state_dict()
            if curriculum.advance():
                # The completed stage snapshot stores its original stage for correct replay.
                state = curriculum.state_dict()
                curriculum.load_state_dict(old_curriculum)
                save(iteration, 'stage_'+old_stage+'.pt')
                curriculum.load_state_dict(state)
                seed_buffers()
                print(f'Curriculum promoted: {old_stage} -> {curriculum.stage}', flush=True)
            if iteration % options.save_interval == 0 or stop:
                save(iteration)
            if stop:
                break
        save(iteration)
    finally:
        pool.shutdown()
        writer.close()
        if wandb_run:
            wandb_run.finish()
    return out/'last.pt'


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--iterations', type=int, default=20000, help='Additional updates when resuming')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--learning-rate',type=float,help='Explicit actor/critic LR override, including on resume')
    p.add_argument('--gamma',type=float,help='Discount per control step; resume restores checkpoint value')
    p.add_argument('--gae-lambda',type=float,help='GAE trace parameter; resume restores checkpoint value')
    p.add_argument('--history-frames',type=int,help='Causal actor history length; 0 keeps the original actor')
    p.add_argument('--history-bins',type=int,help='Ordered temporal pooling bins; defaults to 5')
    p.add_argument('--amp-coef',type=float,help='AMP imitation coefficient; 0 disables AMP')
    p.add_argument('--amp-motion-path',help='Q1 motion CSV directory; defaults to shared/q1-mujoco/resources/motions/q1')
    p.add_argument('--match-motion-constraints',action=argparse.BooleanOptionalAction,default=None,
                   help='Contact/posture constraints in the AMP group; resume restores checkpoint mode')
    p.add_argument('--match-gait-reward',action=argparse.BooleanOptionalAction,default=None,
                   help='Enable match gait/style rewards; resume restores the saved setting')
    p.add_argument('--episode-steps', type=int, help='Override match length, primarily for timeout smoke tests')
    p.add_argument('--device', default='cuda:0', help='Warp physics and learner share this CUDA device')
    p.add_argument('--backend', choices=('warp', 'cpu'), default='warp')
    p.add_argument('--nconmax', type=int, default=768, help='Average contact capacity per 8-robot world')
    p.add_argument('--njmax', type=int, default=3072, help='Constraint capacity per world')
    p.add_argument('--stage', choices=tuple(dict.fromkeys((*COURSES,*STAGES))), default='balance')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint')
    p.add_argument('--curriculum-mode',choices=('serial','parallel'),help='Parallel requires Warp and per-course prerequisite gates')
    p.add_argument('--marching-target-observation',choices=('legacy','origin'),help='Fresh runs use origin; resume retains checkpoint input semantics')
    p.add_argument('--policy-sharing',choices=('shared','independent'),help='Default shared for fresh runs; resume uses checkpoint layout')
    p.add_argument('--start-course', help='Override checkpoint course by name or zero-based index; reset course gates')
    p.add_argument('--save-interval', type=int, default=50)
    p.add_argument('--output', default='logs/train')
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--run-name')
    p.add_argument('--job-type', choices=('train', 'smoke'), default='train')
    return p


if __name__ == '__main__':
    print(run(parser().parse_args()))
