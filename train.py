"""Explicitly gated training entry point. --smoke is bounded and stays offline."""
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml

from q1_supermimic import ROOT
from q1_supermimic.environment import load_environment
from q1_supermimic.learning import SoccerAgent, TemporalReplay
from q1_supermimic.safety import SafetyGate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/soccer.yaml')
    gate = p.add_mutually_exclusive_group(required=True)
    gate.add_argument('--preflight', action='store_true', help='4 physics steps, no learning')
    gate.add_argument('--smoke', action='store_true', help='2 tiny learning iterations (CPU default), no W&B')
    gate.add_argument('--start-training', action='store_true', help='explicitly authorize a training run')
    p.add_argument('--device')
    p.add_argument('--num-envs', type=int)
    p.add_argument('--iterations', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--migrate-curriculum', action='store_true', help='Allow curriculum-only changes to a v3 checkpoint')
    p.add_argument('--no-wandb', action='store_true')
    return p


def build_agent(state_dim, cfg):
    agent = SoccerAgent(state_dim, width=cfg['width'], horizon=cfg['imagination_horizon'],
                        lr=cfg['learning_rate']).to(cfg['device'])
    if 'task_layout' in cfg:
        agent.configure_task(cfg['task_layout'])
    return agent


CURRICULUM_FIELDS = ('balance_release_iteration', 'soccer_progress', 'recent_success_rate',
                     '_curriculum_iteration', '_curriculum_last_adjustment',
                     '_curriculum_quality_window', 'survival_rate_5s',
                     '_survival_pass_mass', '_survival_trial_mass')


def save_checkpoint(path, agent, env, cfg, iteration):
    optimizers = [agent.model_optimizer, agent.actor_optimizer, agent.critic_optimizer]
    payload = {'version': 3, 'iteration': iteration, 'config': cfg,
               'state_dim': agent.dim, 'task_signature': env.task_signature,
               'near_start': env.env.near_start_state(),
               'environment': env.cfg, 'agent': agent.state_dict(),
               'safety': agent.safety_gate.state_dict() if hasattr(agent, 'safety_gate') else None,
               'optimizers': [o.state_dict() for o in optimizers],
               'curriculum': {k: getattr(env.env, k) for k in CURRICULUM_FIELDS},
               'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
               'python_rng': random.getstate(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def validate_checkpoint(checkpoint, env, migrate_curriculum=False):
    if migrate_curriculum:
        keys = ('rewards', 'init_state', 'control', 'randomization', 'asset')
        if any(checkpoint.get('environment', {}).get(k) != env.cfg[k] for k in keys):
            raise ValueError('Curriculum migration cannot change physics, controls or rewards')
        checkpoint = dict(checkpoint, task_signature=env.task_signature)

    if (checkpoint.get('version') != 3 or checkpoint.get('state_dim') != env.state_dim
            or checkpoint.get('task_signature') != env.task_signature):
        raise ValueError('Incompatible task/state schema: use a fresh run for the corrected learning objective; '
                         'old smoke and PPO checkpoints cannot be resumed')


def main():
    args = parser().parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ('device', 'num_envs', 'iterations', 'output'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if key == 'output' else value
    if args.smoke:
        cfg.update(device=args.device or 'cpu', num_envs=2, iterations=2, warmup_steps=8,
                   replay_steps=16, collect_steps=4, model_sequence=2,
                   model_batches=1, batch_size=4, width=32,
                   imagination_horizon=2, actor_start_iteration=0, evaluation_interval=1)
    if args.preflight:
        cfg['num_envs'] = args.num_envs or 2
    if min(cfg[k] for k in ('num_envs', 'iterations', 'collect_steps', 'warmup_steps',
                           'replay_steps', 'model_sequence', 'model_batches', 'batch_size',
                           'width', 'imagination_horizon', 'save_interval')) <= 0:
        raise ValueError('Counts and sizes must be positive')
    if cfg['imagination_horizon'] > cfg['model_sequence']:
        raise ValueError('Imagination horizon must not exceed supervised model sequence')
    if cfg['model_sequence'] > min(cfg['warmup_steps'], cfg['replay_steps']):
        raise ValueError('model_sequence must fit warmup and replay')
    if args.start_training and cfg['num_envs'] % 4:
        raise ValueError('Formal training needs num_envs divisible by four for exact 25/25/50 quotas')
    if cfg['device'].startswith('cuda'):
        # Torch cap excludes Warp allocations; bound physics with num_envs too.
        torch.cuda.set_per_process_memory_fraction(cfg.get('torch_memory_fraction', .30), cfg['device'])
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    env = load_environment(cfg['device'], cfg['num_envs'])
    e = env.env
    cfg['task_layout'] = dict(physical_start=170, physical_end=170+e.model.nq+e.model.nv,
                              root_xy=170+e.root_addr, root_z=172+e.root_addr,
                              root_quat=173+e.root_addr, root_velocity=170+e.model.nq+e.root_vel_addr,
                              joints_start=170+int(e.pos_addrs[0]),
                              ball_start=170+e.ball_addr, default_start=368,
                              progress_index=env.state_dim-16, dt=e.dt,
                              productive_survival=e.cfg['curriculum'].get('productive_survival', False),
                              near_approach_gain=e.cfg['curriculum'].get('near_approach_gain', 1.),
                              stall_penalty_factor=e.cfg['curriculum'].get('stall_penalty_factor', 1.),
                              nominal_height=e.cfg['init_state']['pos'][2],
                              dynamic_ranges=[(0,262),(308,368),
                                              (env.state_dim-18,env.state_dim-16),
                                              (env.state_dim-15,env.state_dim-5)])
    cfg['task_layout'] = json.loads(json.dumps(cfg['task_layout'], default=lambda x: x.item()))
    cfg['learner_version'] = 3
    cfg['task_signature'] = env.task_signature
    cfg['reward_version'] = env.cfg['rewards']['version']
    cfg['scene_sha256'] = hashlib.sha256(Path(env.cfg['asset']['mujoco_file']).read_bytes()).hexdigest()
    if args.preflight:
        for _ in range(4):
            state, target, reward, info = env.step(torch.zeros(cfg['num_envs'], 22, device=cfg['device']))
            assert torch.isfinite(state).all() and torch.isfinite(target).all()
        print(json.dumps({'preflight': 'passed', 'backend': env.cfg['sim']['physics_engine'],
                          'state_dim': env.state_dim, 'actions': 22, 'steps': 4}))
        return
    start = 0
    checkpoint = None
    if args.checkpoint:
        # Checkpoints contain optimizer/RNG state. Load only trusted local files.
        checkpoint = torch.load(args.checkpoint, map_location=cfg['device'], weights_only=False)
        validate_checkpoint(checkpoint, env, args.migrate_curriculum)
        for key in ('width', 'imagination_horizon', 'learning_rate'):
            cfg[key] = checkpoint['config'][key]
        start = checkpoint['iteration']
        cfg['resumed_from'] = str(args.checkpoint.resolve())
        cfg['curriculum_migration'] = args.migrate_curriculum
        if args.smoke:
            cfg['iterations'] = start + 2
        for key, value in checkpoint['curriculum'].items():
            setattr(env.env, key, value)
        env.env.restore_near_start_state(checkpoint['near_start'])
        env.env.set_training_iteration(start)
        env.env.reset()
    agent = build_agent(env.state_dim, cfg)
    agent.make_optimizers()
    if checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        for optimizer, state in zip((agent.model_optimizer, agent.actor_optimizer,
                                     agent.critic_optimizer), checkpoint['optimizers']):
            optimizer.load_state_dict(state)
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if checkpoint['cuda_rng'] is not None and cfg['device'].startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    if cfg['iterations'] <= start:
        raise ValueError('iterations is a cumulative target and must exceed checkpoint iteration')
    output = Path(cfg['output']).resolve() / datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg))
    (output / 'environment.yaml').write_text(yaml.safe_dump(env.cfg))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(output / 'summaries'))
    wandb_run = None
    if args.start_training and not args.no_wandb:
        import wandb
        wandb_run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
                              config={**cfg, 'environment': env.cfg, 'algorithm': 'model_based_soccer_with_real_policy_gradient',
                                      'reward_includes_amp': False}, dir=str(output), name=output.name)
        wandb_run.define_metric('iteration')
        wandb_run.define_metric('*', step_metric='iteration')
    replay = TemporalReplay(cfg['replay_steps'], cfg['num_envs'], env.state_dim, cfg['device'])
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    state = env.state()
    latest_rollout = None

    def collect(steps, record_policy=True):
        nonlocal state, latest_rollout
        totals = dict(control_reward=0., episodes=0., falls=0., successes=0.)
        feedback = None
        on_policy = []
        with torch.no_grad():
            for _ in range(steps):
                noise = cfg['exploration_std'] + cfg.get('exploration_progress_scale', .03) * env.env.soccer_progress
                mean = agent.actor(agent.normalize(state)[:, :agent.obs_dim])
                action = agent.action_limit * (mean + noise / agent.action_limit * torch.randn(
                    cfg['num_envs'], 22, device=cfg['device'])).tanh()
                following, target, reward, info = env.step(action)
                replay.add(state, action, target, reward, info['done'], info['fall'])
                if record_policy:
                    on_policy.append((state, action, target, reward, info['done'].float(), info['fall'].float()))
                state = following
                values = torch.stack(list(info['metrics'].values()))
                feedback = values if feedback is None else feedback + values
                totals['policy/action_abs_mean'] = totals.get('policy/action_abs_mean', 0.) + action.abs().mean().item() / steps
                totals['policy/action_saturation_fraction'] = totals.get('policy/action_saturation_fraction', 0.) + (action.abs() > .95 * agent.action_limit).float().mean().item() / steps
                totals['control_reward'] += reward.mean().item() / steps
                totals['episodes'] += info['done'].sum().item()
                totals['falls'] += info['fall'].sum().item()
                totals['successes'] += info['success'].sum().item()
        totals.update(dict(zip(info['metrics'], (feedback / steps).cpu().tolist())))
        totals.update(env.episode_metrics.drain(env.env.dt))
        if record_policy:
            latest_rollout = tuple(torch.cat([transition[i] for transition in on_policy], 0) for i in range(6))
        return totals

    try:
        # Replay and simulated trajectories restart on resume; statistics and
        # optimizer states persist. This is not bitwise trajectory continuation.
        collect(cfg['warmup_steps'], record_policy=False)
        if not checkpoint:
            agent.calibrate(replay.states[:replay.count].flatten(0, 1))
        gate = SafetyGate(agent, count=4 if args.smoke else cfg.get('evaluation_envs', 32),
                          steps=20 if args.smoke else cfg.get('evaluation_steps', 300))
        if checkpoint and checkpoint.get('safety'):
            gate.load_state_dict(checkpoint['safety'])
        agent.safety_gate = gate
        print('Real-physics neutral baseline: ' + json.dumps(gate.baseline), flush=True)
        if args.migrate_curriculum and not args.smoke:
            initial_evaluation = gate.check(agent, env.env.soccer_progress)
            print('Migrated policy evaluation: ' + json.dumps(initial_evaluation), flush=True)
            save_checkpoint(output / f'checkpoint_{start}.pt', agent, env, cfg, start)
            (output / 'last.pt').symlink_to(f'checkpoint_{start}.pt')
        for iteration in range(start + 1, cfg['iterations'] + 1):
            began = time.monotonic()
            if args.smoke and not checkpoint and iteration == 2:
                # Exercise both standing and full soccer feedback in a bounded check.
                env.env.balance_release_iteration = 0
                env.env.soccer_progress = 1.
                env.env.set_training_iteration(iteration)
                env.env.reset()
                state = env.state()
                replay.count = replay.cursor = 0  # Manual smoke reset starts a new sequence.
            env.env.set_training_iteration(iteration)
            agent.action_limit.fill_(.10 + .90 * env.env.soccer_progress)
            totals = collect(cfg['collect_steps'])
            losses = [agent.fit_model(replay.sample(cfg['batch_size'], cfg['model_sequence']))
                      for _ in range(cfg['model_batches'])]
            critic_loss = agent.fit_critic(replay.sample(cfg['batch_size'], cfg['model_sequence']))
            actor_loss = 0.
            model_check = agent.validate_model(replay.sample(cfg['batch_size'], cfg['imagination_horizon'], validation=True))
            ready = (model_check['model_validation/physical_mse'] < cfg.get('model_max_mse', 1.)
                     and model_check['model_validation/persistence_ratio'] < cfg.get('model_max_ratio', .8)
                     and model_check['model_validation/height_mae_m'] < cfg.get('model_max_height_mae', .01)
                     and model_check['model_validation/joint_rmse_rad'] < cfg.get('model_max_joint_rmse', .025)
                     and model_check['model_validation/root_velocity_rmse'] < cfg.get('model_max_velocity_rmse', .5))
            if iteration >= cfg['actor_start_iteration'] and (ready or args.smoke):
                initial = replay.sample(cfg['batch_size'], 1)[0][:, 0]
                indices = torch.randperm(len(latest_rollout[0]), device=cfg['device'])[:4096]
                actor_loss, _ = agent.improve(initial, tuple(x[indices] for x in latest_rollout),
                                              cfg['exploration_std'] + cfg.get('exploration_progress_scale', .03) * env.env.soccer_progress)
            evaluation = {}
            if iteration % cfg.get('evaluation_interval', 50) == 0 or iteration == cfg['iterations']:
                evaluation = gate.check(agent, env.env.soccer_progress)
                print('Real-physics policy check: ' + json.dumps(evaluation), flush=True)
            metrics = {'iteration': iteration, 'reward/control_step': totals['control_reward'],
                       'loss/model': float(np.mean(losses)), 'loss/actor': actor_loss,
                       'loss/critic': critic_loss, 'curriculum/soccer_progress': env.env.soccer_progress,
                       'stability/active_over_5s': (env.env.episode_length_buf * env.env.dt >= 5).float().mean().item(),
                       'stability/reset_cohort_survival_5s': env.env.survival_rate_5s,
                       'stability/base_height_mean': env.env.base_pos[:, 2].mean().item(),
                       'curriculum/balance_phase': float(env.env.balance_phase),
                       'curriculum/full_soccer_progress': env.env.full_soccer_progress,
                       'curriculum/near_soccer_progress': max(env.env.soccer_progress, env.cfg['curriculum'].get('near_task_min_progress', 0.)) if not env.env.balance_phase else 0.,
                       'curriculum/reference_init_probability': env.env.reference_init_probability,
                       'episodes/completed': totals['episodes'],
                       'performance/env_steps_per_second': cfg['collect_steps'] * cfg['num_envs'] / (time.monotonic() - began)}
            metrics.update(model_check)
            metrics.update(agent.last_diagnostics)
            metrics.update(evaluation)
            metrics['learning/model_ready'] = float(ready)
            metrics['learning/action_limit'] = float(agent.action_limit)
            metrics.update({k: v for k, v in totals.items() if '/' in k or k in ('steps', 'success_rate', 'fall_rate')})
            if 'episode/duration_s' in totals:
                metrics['reward'] = totals['reward']
            metrics.update({'task/reward_version': env.cfg['rewards']['version'],
                            'curriculum/near_start_progress': env.env.near_start_progress,
                            'curriculum/near_start_fraction': env.env.near_start_fraction,
                            'curriculum/near_start_trials': env.env.near_start_trials})
            if env.env.near_start_trials:
                metrics['curriculum/near_start_success_rate'] = env.env.near_start_success_rate
            if totals['episodes']:
                metrics.update({'instant/fall_rate': totals['falls'] / totals['episodes'],
                                'instant/success_rate': totals['successes'] / totals['episodes']})
            if cfg['device'].startswith('cuda'):
                metrics['performance/cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                metrics['performance/cuda_reserved_gib'] = torch.cuda.memory_reserved() / 2**30
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError('Non-finite iteration metrics')
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            for key, value in metrics.items():
                writer.add_scalar(key, value, iteration)
            writer.flush()
            if wandb_run:
                wandb_run.log(metrics)
            print(json.dumps(metrics), flush=True)
            if iteration % cfg['save_interval'] == 0 or iteration == cfg['iterations'] or stopped:
                path = output / f'checkpoint_{iteration}.pt'
                save_checkpoint(path, agent, env, cfg, iteration)
                link = output / 'last.pt'
                link.unlink(missing_ok=True)
                link.symlink_to(path.name)
            if stopped:
                break
    finally:
        writer.close()
        if wandb_run:
            wandb_run.finish()
    print(f'Run saved: {output}', flush=True)


if __name__ == '__main__':
    main()
