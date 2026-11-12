"""Batched Warp + official MAPPO updates; no CPU rollout transfers."""
from pathlib import Path
import json
import signal
import os
import subprocess
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .course_env import CourseWarpEnv
from .gpu_learning import GpuRollout
from .learning import make_args, make_team, save_team, load_team, policy_sharing, with_team_identity, MARCHING_EXPLORATION_REVISION, MATCH_EXPLORATION_REVISION, MarchingGaussian, configure_temporal
from .course_config import make_curriculum, LOW_LEVEL, DURATIONS, COURSES, GAIT_COURSES, BALL_TASK_REVISION, CONTESTS, marching_observation_mode, FULL_MATCHES, MATCH_TASK_REVISION
from .distributed import Distributed
from .gait import METRIC_NAMES, REWARD_REVISION
from .mixed_env import MixedCourseEnv, finish_course_step
from .telemetry import SKILL_NAMES, active_team_reward, course_metrics, macro_metrics, match_metrics
from .course_table import COURSE_TABLE_KEY,course_table_html
from .match_task import MatchTask,MatchControl
from .history import ActorHistory,history_payload,configure_history
from .match_motion import configure_motion


def run(options):
    if min(options.num_envs, options.horizon, options.epochs, options.iterations, options.save_interval) < 1:
        raise ValueError('Training sizes must be positive')
    if options.episode_steps is not None and options.episode_steps < 1:
        raise ValueError('episode-steps must be positive')
    device = torch.device(options.device)
    if device.type != 'cuda':
        raise ValueError('--backend warp requires --device cuda:0')
    distributed = Distributed(device)
    try:return _run(options,device,distributed)
    finally:distributed.close()


def _run(options,device,distributed):
    torch.set_num_threads(2)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    args = make_args(options.num_envs, options.horizon, options.epochs)
    args.marching_exploration_revision=MARCHING_EXPLORATION_REVISION
    args.match_exploration_revision=MATCH_EXPLORATION_REVISION
    if options.learning_rate is not None:
        if options.learning_rate<=0:raise ValueError('learning-rate must be positive')
        args.lr=args.critic_lr=options.learning_rate
    checkpoint = torch.load(options.checkpoint,map_location=device,weights_only=False) if options.checkpoint else None
    motion_config,motion_change=configure_motion(options,checkpoint,checkpoint['iteration'] if checkpoint else 0)
    history_change=configure_history(args,options,checkpoint)
    saved_amp=checkpoint.get('amp',{}) if checkpoint else {}
    from .amp import AMP_REVISION
    if getattr(options,'amp_coef',None) is None:options.amp_coef=saved_amp.get('coefficient',0.)
    if options.amp_coef<0:raise ValueError('AMP coefficient must be nonnegative')
    args.amp_groups=bool(options.amp_coef)
    amp_change=bool(checkpoint and (saved_amp.get('coefficient',0.)!=options.amp_coef or
                    (options.amp_coef and (saved_amp.get('revision',1)!=AMP_REVISION or not checkpoint['args'].get('amp_groups',False)))))
    if not getattr(options,'amp_motion_path',None):
        from .scene import MUJOCO_ROOT
        options.amp_motion_path=str(MUJOCO_ROOT/'resources/motions/q1')
    saved_gait=checkpoint.get('match_gait_reward',True) if checkpoint else True
    if getattr(options,'match_gait_reward',None) is None:options.match_gait_reward=saved_gait
    gait_reward_change=checkpoint is not None and options.match_gait_reward!=saved_gait
    temporal_change=configure_temporal(args,options,checkpoint)
    recovery_start=checkpoint.get('marching_recovery_start',checkpoint['iteration']) if checkpoint else 0
    args.marching_exploration_mix=MarchingGaussian.mixing_at(checkpoint['iteration'] if checkpoint else 0,recovery_start)
    options.curriculum_mode = getattr(options,'curriculum_mode',None) or (checkpoint.get('curriculum_mode','serial') if checkpoint else 'serial')
    options.marching_target_observation=marching_observation_mode(getattr(options,'marching_target_observation',None),checkpoint)
    parallel = options.curriculum_mode=='parallel'
    curriculum=make_curriculum(options.curriculum_mode,options.stage,checkpoint,options.start_course)
    if temporal_change:curriculum.recertify('temporal_change_recertification')
    if history_change:curriculum.recertify('history_encoder_recertification')
    if amp_change:curriculum.recertify('amp_change_recertification')
    if motion_change:
        if parallel:
            for name in FULL_MATCHES:
                curriculum.courses[name].recertify('motion_constraints_recertification')
                curriculum.mastered.discard(name)
        elif curriculum.stage in FULL_MATCHES:curriculum.recertify('motion_constraints_recertification')
    start=checkpoint['iteration'] if checkpoint else 0
    direct_start=curriculum.stage if options.start_course is not None else checkpoint.get('direct_start_course') if checkpoint else None
    task_switch=checkpoint is not None and options.start_course is not None and curriculum.stage!=checkpoint['curriculum']['stage']
    match_change=checkpoint is not None and checkpoint.get('match_task_revision',0)!=MATCH_TASK_REVISION
    if match_change:
        if parallel:
            for name in FULL_MATCHES:
                curriculum.courses[name].recertify('continuous_match_recertification')
                curriculum.mastered.discard(name)
        elif curriculum.stage in FULL_MATCHES:curriculum.recertify('continuous_match_recertification')
    if checkpoint and checkpoint.get('reward_version',1) != REWARD_REVISION:
        curriculum.recertify('reward_change_recertification')
    if checkpoint and checkpoint.get('ball_task_revision',0)!=BALL_TASK_REVISION:
        if parallel:
            for name,c in curriculum.courses.items():
                if name not in LOW_LEVEL:
                    c.recertify('single_ball_task_recertification')
                    curriculum.mastered.discard(name)
        elif curriculum.stage not in LOW_LEVEL:
            curriculum.recertify('single_ball_task_recertification')
    shared = policy_sharing(options, checkpoint)
    groups = [slice(None)] if shared else [slice(0,4),slice(4,8)]
    agents = 8 if shared else 4
    if parallel:
        if not shared:raise ValueError('Parallel curricula require a shared policy')
        env = MixedCourseEnv(options.num_envs,str(device),options.seed,curriculum.allocation(options.num_envs),
                             nconmax=options.nconmax,njmax=options.njmax,episode_steps=options.episode_steps,marching_target_observation=options.marching_target_observation)
    else:
        env = CourseWarpEnv(options.num_envs, str(device), options.seed, curriculum.stage,
                            nconmax=options.nconmax, njmax=options.njmax,marching_target_observation=options.marching_target_observation)
    def match_groups():
        if parallel:return [(name,group) for name,(_,group) in env.groups.items() if name in FULL_MATCHES]
        return [(env.course,env)] if env.course in FULL_MATCHES else []
    def restore_match_controls():
        saved=checkpoint.get('match_controls',{}) if checkpoint and not temporal_change else {}
        for name,group in match_groups():
            state=saved.get(name)
            if match_change and state is not None:
                state={'amplitude':state['amplitude'],'level':state.get('level',0)}
            if amp_change and state is not None:
                state={'amplitude':state['amplitude'],'level':state.get('level',0)}
            if motion_change and state is not None:
                state={'amplitude':state['amplitude'],'level':state.get('level',0)}
            if gait_reward_change and state is not None:
                state={'amplitude':state['amplitude'],'level':state.get('level',0)}
            group.match_task.control=MatchControl(state)
            group.match_task.gait_reward_enabled=options.match_gait_reward
            group.course_amplitude=group.match_task.control.amplitude
            group.match_task.discount=args.gamma
            if args.gamma!=group.match_task.discount:raise ValueError('Match potential and PPO discounts differ')
    restore_match_controls()
    uses_match=bool(match_groups())
    if gait_reward_change:
        for name,_ in match_groups():
            if parallel:curriculum.courses[name].recertify('match_gait_reward_change')
            else:curriculum.recertify('match_gait_reward_change')
    teams = [make_team(args, env, device, gpu_rollout=True,shared=shared) for _ in groups]
    if checkpoint:
        if checkpoint.get('backend') != 'warp':
            raise ValueError('Use a Warp checkpoint for continuation; CPU legacy runs are reference-only')
        if len(checkpoint['teams']) != len(teams):
            raise ValueError('Checkpoint policy count does not match policy sharing')
        for state, (policy, trainer, _) in zip(checkpoint['teams'], teams):
            load_team(state, policy, trainer,match_features=uses_match and not parallel)
            if checkpoint.get('reward_version',1) != REWARD_REVISION or task_switch or (match_change and uses_match) or temporal_change or amp_change or (motion_change and uses_match) or (gait_reward_change and uses_match):
                trainer.value_normalizer.running_mean.zero_()
                trainer.value_normalizer.running_mean_sq.zero_()
                trainer.value_normalizer.debiasing_term.zero_()
                if hasattr(trainer,'amp_value_normalizer'):
                    for value in trainer.amp_value_normalizer.buffers():value.zero_()
        states = checkpoint.get('distributed_rng', [])
        state = states[distributed.rank] if len(states) == distributed.world_size else checkpoint
        if len(states) == distributed.world_size or distributed.primary:
            torch.set_rng_state(state['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            env.generator.set_state(state['env_rng'].cpu())
            np.random.set_state(state['numpy_rng'])
        else:
            # Expanding a single-GPU checkpoint must not duplicate every rollout.
            seed = options.seed + start + 100003*distributed.rank
            torch.manual_seed(seed)
            env.generator.manual_seed(seed)
            np.random.seed(seed)
        if options.start_course is not None:
            env.generator.manual_seed(options.seed+start+100003*distributed.rank)
    elif not distributed.primary:
        seed = options.seed + 100003*distributed.rank
        torch.manual_seed(seed)
        env.generator.manual_seed(seed)
        np.random.seed(seed)
    if parallel:
        saved_rng=state.get('course_rng',{}) if checkpoint and (len(states)==distributed.world_size or distributed.primary) else {}
        env.restore_course_rng(saved_rng)
    for policy, _, _ in teams:
        distributed.attach(policy)
    if options.learning_rate is not None:
        for policy,_,_ in teams:
            for opt in (policy.actor_optimizer,policy.critic_optimizer):
                for group in opt.param_groups:group['lr']=options.learning_rate
    raw_obs_dim=env.observation_space.shape[0]
    obs_dim=raw_obs_dim+(2 if shared else 0)
    state_dim=env.share_observation_space.shape[0]+(2 if shared else 0)
    rollout = GpuRollout(options.horizon, options.num_envs, device, obs_dim=raw_obs_dim,amp_groups=args.amp_groups)
    history=ActorHistory(options.num_envs,8,raw_obs_dim,args.history_frames,args.history_bins,device) if args.history_frames else None
    if history is not None:
        rollout.history=torch.zeros((options.horizon+1,options.num_envs,8,args.history_bins,raw_obs_dim),device=device,dtype=torch.float16)
    amp=None
    if options.amp_coef:
        from .amp import MotionPrior
        amp=MotionPrior(options.amp_motion_path,device,options.amp_coef,start,saved_amp or None)
    rnn = torch.zeros((options.num_envs*agents, 1, args.hidden_size), device=device)
    ones = torch.ones((options.num_envs*agents, 1), device=device)
    owners = (torch.arange(8,device=device)//4)[None]

    def policy_inputs(obs,state):
        if shared:
            return with_team_identity(obs,owners),with_team_identity(state,owners,role_suffix=4)
        return obs,state
    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=True)
    if distributed.primary:
        config = dict(vars(options), physics_config=env.physics_config,world_size=distributed.world_size, per_course_metrics=True, ball_task_revision=BALL_TASK_REVISION,
                      total_num_envs=options.num_envs*distributed.world_size)
        (out/'config.json').write_text(json.dumps(config, indent=2))
    writer = SummaryWriter(str(out/'tensorboard')) if distributed.primary else None
    wandb_run = None
    stop = False

    def stop_handler(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def reset_stage():
        if parallel:
            env.configure(curriculum.allocation(options.num_envs))
            rollout.course_ids=env.course_ids
        else:
            env.configure(curriculum.stage)
            env.episode_steps = options.episode_steps or DURATIONS[curriculum.stage]
        for _,group in match_groups():
            group.match_task.discount=args.gamma
            group.match_task.gait_reward_enabled=options.match_gait_reward
            group.match_task.motion.enabled=motion_config['enabled']
            group.match_task.motion.start_iteration=motion_config['start_iteration']
            group.match_task.motion.iteration=start
        obs, _ = env.reset()
        rollout.obs[0].copy_(obs)
        if history is not None:
            history.data.zero_();history.cursor=0;history.push(obs)
            rollout.history[0].copy_(history.pooled())
        # Encourage renewed exploration while learning to lift from a stable stance.
        for _, trainer, _ in teams:
            trainer.entropy_coef = env.entropy_coef if parallel else .005 if env.course=='step_in_place' else args.entropy_coef

    last_saved=start

    def save(iteration, name='last.pt'):
        nonlocal last_saved
        rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   env_rng=env.generator.get_state(), numpy_rng=np.random.get_state())
        if parallel:rng['course_rng']=env.course_rng()
        rank_rng = distributed.objects(rng)
        difference = distributed.check_parameters(teams)
        amp_difference=amp.parameter_difference() if amp is not None else 0.
        if name=='last.pt':last_saved=iteration
        if not distributed.primary:
            return
        payload = dict(version=4, physics_config=env.physics_config,match_task_revision=MATCH_TASK_REVISION,
                       motion_constraints=motion_config,
                       match_gait_reward=options.match_gait_reward,
                       match_exploration_revision=MATCH_EXPLORATION_REVISION,
                       match_controls={name:group.match_task.control.state_dict() for name,group in match_groups()},
                       direct_start_course=direct_start, marching_recovery_start=recovery_start, marching_exploration_mix=args.marching_exploration_mix, marching_exploration_revision=args.marching_exploration_revision, marching_target_observation=options.marching_target_observation, reward_version=REWARD_REVISION, ball_task_revision=BALL_TASK_REVISION, obs_dim=obs_dim, raw_obs_dim=raw_obs_dim,
                       policy_sharing=options.policy_sharing, curriculum_mode=options.curriculum_mode, backend='warp', iteration=iteration, args=vars(args),
                       teams=[save_team(p, t) for p, t, _ in teams], curriculum=curriculum.state_dict(),
                       **rng, distributed_rng=rank_rng, world_size=distributed.world_size,
                       parameter_max_difference=difference,amp_parameter_max_difference=amp_difference,
                       actor_frame_dim=obs_dim,actor_history_frames=args.history_frames,actor_history_bins=args.history_bins)
        if amp is not None:payload['amp']=amp.checkpoint()
        temp = out/(name+'.tmp')
        torch.save(payload, temp)
        temp.replace(out/name)

    reset_stage()
    iteration = start
    exit_code=0
    try:
        if options.wandb and distributed.primary:
            import wandb
            wandb_run = wandb.init(entity='mohismlab-robotics', project='q1-humanoid-kick-4v4',
                name=options.run_name or out.name, job_type=options.job_type,
                tags=['q1', '4v4', 'mappo', 'warp', options.job_type], config=vars(options),
                dir=str(out.resolve()), settings=wandb.Settings(init_timeout=60))
            wandb_run.config.update(dict(vars(options), world_size=distributed.world_size,
                total_num_envs=options.num_envs*distributed.world_size,
                reward_revision=REWARD_REVISION, match_task_revision=MATCH_TASK_REVISION,
                physics_config=env.physics_config,
                match_exploration_revision=MATCH_EXPLORATION_REVISION,
                marching_exploration_revision=args.marching_exploration_revision, curriculum_window_policy='per_team', per_course_metrics=True, ball_task_revision=BALL_TASK_REVISION), allow_val_change=True)
            wandb_run.define_metric('iteration')
            wandb_run.define_metric('*', step_metric='iteration')
            wandb_run.define_metric('debug/*', hidden=True)
            (out/'wandb_run.json').write_text(json.dumps({'id': wandb_run.id, 'url': wandb_run.url}))
        distributed.maximum(0)  # All workers wait for primary W&B initialization.
        for iteration in range(start+1, start+options.iterations+1):
            for _,group in match_groups():group.match_task.motion.iteration=iteration
            if amp is not None:
                amp.iteration=iteration
                args.amp_advantage_weight=min(1.,max(0.,(iteration-amp.start_iteration)/200))
            args.marching_exploration_mix=MarchingGaussian.mixing_at(iteration,recovery_start)
            for policy,_,_ in teams:policy.actor.act.action_out.mean_mix=args.marching_exploration_mix
            torch.cuda.synchronize(device)
            tic = time.perf_counter()
            completed = []
            raw_reward = torch.zeros(2, device=device)
            skill_stats = torch.zeros(len(SKILL_NAMES),device=device)
            gait_stats = torch.zeros(len(METRIC_NAMES),device=device)
            match_stats=torch.zeros((len(COURSES),len(MatchTask.metrics)),device=device)
            reward_terms=torch.zeros((len(COURSES),len(MatchTask.terms),2),device=device)
            if parallel:env.begin_rollout()
            for step in range(options.horizon):
                rollout.active[step].copy_(env.active)
                obs = rollout.obs[step]
                state = env.state(obs)
                with torch.no_grad():
                    actor_obs,critic_state=policy_inputs(obs,state)
                    actor_obs=history_payload(actor_obs,rollout.history[step] if history is not None else None,shared,owners)
                    for t, (policy, trainer, _) in enumerate(teams):
                        sl = groups[t]
                        trainer.prep_rollout()
                        value, action, logprob, _, _ = policy.get_actions(critic_state[:, sl].reshape(-1, state_dim),
                            actor_obs[:, sl].reshape(-1, actor_obs.shape[-1]), rnn, rnn, ones)
                        rollout.actions[step, :, sl].copy_(action.reshape(options.num_envs, agents, 22))
                        rollout.log_probs[step, :, sl].copy_(logprob.reshape(options.num_envs, agents, 1))
                        rollout.values[step, :, sl].copy_(trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1))
                        if args.amp_groups:
                            rollout.amp_values[step,:,sl].copy_(trainer.amp_value_normalizer.denormalize(policy.critic.v_out.amp_value).reshape(options.num_envs,agents,1))
                    if amp is not None:amp.begin_step(match_groups())
                    terminal_obs, terminal_state, reward, terminated, truncated, info = env.step(rollout.actions[step])
                    if args.amp_groups:
                        style_reward=torch.zeros_like(reward)
                        rollout.amp_active[step].zero_()
                        for name,group in match_groups():
                            sl=env.groups[name][0] if parallel else slice(None)
                            style_reward[sl,:,0]=group.match_task.style_reward
                            rollout.amp_active[step,sl]=group.active
                        reward=reward-style_reward
                        rollout.amp_rewards[step].copy_(style_reward)
                    gait_stats += torch.stack([env.gait_metrics[k] for k in METRIC_NAMES])
                    skill_stats += torch.stack([env.last_stats[k] for k in SKILL_NAMES])
                    raw_reward += active_team_reward(reward,env.active)
                    if not parallel and env.course in FULL_MATCHES:
                        index=COURSES.index(env.course)
                        match_stats[index]+=env.match_task.last_metrics
                        reward_terms[index]+=env.match_task.last_terms
                    # Bootstrap timeout final states BEFORE resetting. No CPU value normalization.
                    _,terminal_state=policy_inputs(terminal_obs,terminal_state)
                    for t, (policy, trainer, _) in enumerate(teams):
                        sl = groups[t]
                        value = policy.get_values(terminal_state[:, sl].reshape(-1, state_dim), rnn, ones)
                        timeout_mask = truncated[:,sl,None] if truncated.ndim==2 else truncated[:,None,None]
                        reward[:, sl] += args.gamma*trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1)*timeout_mask
                        if args.amp_groups:
                            rollout.amp_rewards[step,:,sl]+=args.gamma*trainer.amp_value_normalizer.denormalize(policy.critic.v_out.amp_value).reshape(options.num_envs,agents,1)*timeout_mask
                    rollout.rewards[step].copy_(reward)
                    done = terminated | truncated
                    terminal_mask = done if done.ndim==2 else done[:,None].expand(-1,8)
                    rollout.masks[step].copy_((~terminal_mask)[:,:,None])
                    if parallel:
                        terminal_obs,ended=env.finish_step()
                        if len(ended):completed.append(ended)
                    else:
                        reset_obs,ended=finish_course_step(env,terminated,truncated,info)
                        if reset_obs is not None:terminal_obs=reset_obs
                        if len(ended):completed.append(ended)
                    rollout.obs[step+1].copy_(terminal_obs)
                    if history is not None:
                        history.push(terminal_obs,terminal_mask)
                        rollout.history[step+1].copy_(history.pooled())
            distributed.check_health(env.check)  # All ranks stop before ANY optimizer update on failure.
            with torch.no_grad():
                state = env.state(rollout.obs[-1])
                _,state=policy_inputs(rollout.obs[-1],state)
                for t, (policy, trainer, _) in enumerate(teams):
                    sl = groups[t]
                    value = policy.get_values(state[:, sl].reshape(-1, state_dim), rnn, ones)
                    rollout.values[-1, :, sl].copy_(trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1))
                    if args.amp_groups:
                        rollout.amp_values[-1,:,sl].copy_(trainer.amp_value_normalizer.denormalize(policy.critic.v_out.amp_value).reshape(options.num_envs,agents,1))
                rollout.compute_returns(args.gamma, args.gae_lambda)
            metrics = {}
            if amp is not None:metrics.update(amp.update())
            for t, (_, trainer, _) in enumerate(teams):
                trainer.prep_training()
                stats = rollout.update(None if shared else t, trainer, args, active=rollout.active)
                prefix = 'shared' if shared else ('red','blue')[t]
                metrics.update({f'{prefix}/{k}': v for k, v in stats.items()})
            for t,side in enumerate(('red','blue')):
                metrics[f'{side}/reward'] = (raw_reward[t]/options.horizon).item()
            if not all(np.isfinite(x) for x in metrics.values()):
                raise FloatingPointError('Nonfinite MAPPO update')
            local_episodes = torch.cat(completed) if completed else torch.empty((0,10),device=device)
            episodes = distributed.rows(local_episodes).cpu().tolist()
            if distributed.world_size > 1:
                averaged = distributed.mean(torch.tensor(list(metrics.values()),device=device)).cpu().tolist()
                metrics = dict(zip(metrics,averaged))
                skill_stats = distributed.mean(skill_stats)
                gait_stats = distributed.mean(gait_stats)
            for stable, success, goal, fallen, duration, progress, touch, recover, team, course_id in episodes:
                curriculum.record(dict(stage=COURSES[int(course_id)],stable=stable,success=success,duration=duration,
                                       progress=progress,touch=touch,recover=recover,team=team,goal=goal,fallen=fallen))
            torch.cuda.synchronize(device)
            seconds = distributed.maximum(time.perf_counter()-tic)
            # Torch's allocator excludes Warp, so also record whole-process NVML usage.
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            for line in query.stdout.splitlines():
                fields = line.split(',')
                if len(fields) == 2 and fields[0].strip() == str(os.getpid()):
                    metrics['gpu_process_used_gib'] = float(fields[1].strip())/1024
            memories = distributed.objects(metrics.get('gpu_process_used_gib',0.))
            metrics['gpu_process_used_gib'] = max(memories)
            metrics.update(iteration=iteration, seconds=seconds, backend='warp',
                           policy_sharing=options.policy_sharing, curriculum_mode=options.curriculum_mode,
                           world_size=distributed.world_size, total_num_envs=options.num_envs*distributed.world_size,
                           gpu_total_used_gib=sum(memories), gpu_worker_used_gib=memories,
                           episodes=len(episodes), goals=sum((.5 if COURSES[int(x[9])] in CONTESTS else 1) for x in episodes if x[2]),
                           falls=sum(bool(x[3]) for x in episodes),
                           env_steps_per_second=distributed.world_size*options.num_envs*options.horizon/seconds,
                           agent_steps_per_second=8*distributed.world_size*options.num_envs*options.horizon/seconds,
                           cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                           curriculum=curriculum.metrics())
            cm = curriculum.metrics()
            gm=dict(zip(METRIC_NAMES,(gait_stats/options.horizon).cpu().tolist()))
            metrics['gait']=gm
            ss=dict(zip(SKILL_NAMES,(skill_stats/options.horizon).cpu().tolist()))
            cloud = {'iteration':iteration,
                     '01_总览/课程编号':curriculum.index,'01_总览/阶段成功率':cm['success_rate'],
                     '01_总览/稳定率':cm['stable_rate'],'01_总览/回合时长_秒':cm['episode_seconds'],
                     '01_总览/红队回报':metrics['red/reward'],'01_总览/蓝队回报':metrics['blue/reward'],
                     '02_晋级/连续通过窗口':cm['passed_windows'],'02_晋级/窗口样本数':cm['completed'],
                     '02_晋级/阶段更新次数':cm['stage_updates'],
                     '02_晋级/红队成功率':cm['red_success'],'02_晋级/蓝队成功率':cm['blue_success'],
                     '02_晋级/红队稳定率':cm['red_stability'],'02_晋级/蓝队稳定率':cm['blue_stability'],
                     '02_晋级/红队窗口样本':cm['red_gate_samples'],'02_晋级/蓝队窗口样本':cm['blue_gate_samples'],
                     '02_晋级/红队监测样本':cm['red_samples'],'02_晋级/蓝队监测样本':cm['blue_samples'],
                     'debug/奖励修订':REWARD_REVISION,
                     'debug/课程3探索过渡比例':args.marching_exploration_mix,
                     'debug/熵系数':teams[0][1].entropy_coef,
                     '03_技能/脚部触球率':cm['foot_touch_rate'],'03_技能/触球后恢复率':cm['recovery_rate'],
                     '03_技能/净推进_米':cm['progress_m'],
                     '03_技能/左脚触球':ss['left'],'03_技能/右脚触球':ss['right'],
                     '03_技能/正踢奖励':ss['forward'],'03_技能/侧踢奖励':ss['side'],
                     '01_总览/平均身高_米':ss['height'],'01_总览/回合摔倒率':sum(bool(x[3]) for x in episodes)/max(len(episodes),1),
                     '01_总览/直立程度':ss['upright'],
                     '04_性能/球场步每秒':metrics['env_steps_per_second'],
                     '04_性能/进程显存_GiB':metrics.get('gpu_process_used_gib',0.),
                     '04_性能/GPU数量':distributed.world_size,
                     '04_性能/并行球场总数':options.num_envs*distributed.world_size,
                     '04_性能/GPU总显存_GiB':sum(memories),
                     '阶段名称':curriculum.stage,'晋级状态':cm['gate_reason']}
            if parallel:
                cloud['01_总览/已掌握课程数']=cm['mastered_count']
                cloud['01_总览/已解锁课程数']=cm['unlocked_count']
                cloud['01_总览/主训课程数']=len(cm['ready'])
                cloud.update(macro_metrics(cm['courses'],cm['ready'],cm['mastered']))
                course_gait=(distributed.mean(env.course_stat_sum)/options.horizon).cpu().tolist()
                course_rewards=(distributed.mean(env.course_reward_sum)/options.horizon).cpu().tolist()
                course_skills=(distributed.mean(env.course_skill_sum)/options.horizon).cpu().tolist()
                for name,stats in cm['courses'].items():
                    i=COURSES.index(name); worlds=env.allocations.get(name,0)*distributed.world_size
                    cloud.update(course_metrics(name,stats,course_rewards[i] if worlds else (None,None),
                        dict(zip(METRIC_NAMES,course_gait[i])),dict(zip(SKILL_NAMES,course_skills[i])),
                        worlds,options.num_envs*distributed.world_size,name in curriculum.mastered))
            else:
                cloud.update(course_metrics(curriculum.stage,cm,
                    (metrics['red/reward'],metrics['blue/reward']),gm,ss,
                    options.num_envs*distributed.world_size,options.num_envs*distributed.world_size))
            if direct_start is not None:cloud['01_总览/直接起始课程编号']=COURSES.index(direct_start)
            match_values=(distributed.mean(env.course_match_sum if parallel else match_stats)/options.horizon).cpu().tolist()
            reward_values=(distributed.mean(env.course_reward_terms_sum if parallel else reward_terms)/options.horizon).cpu().tolist()
            def update_match_controls():
                for name,group in match_groups():
                    index=COURSES.index(name)
                    values=dict(zip(MatchTask.metrics,match_values[index]))
                    used_amplitude=group.course_amplitude
                    evidence={key:values[key]*options.horizon*distributed.world_size for key in MatchControl.counters}
                    group.match_task.control.update(match_values[index][:2],match_values[index][2:4],evidence)
                    buckets=distributed.mean(torch.stack([(group.match_task.spawn_bucket==i).float().sum() for i in range(4)]))*distributed.world_size
                    bucket_worlds=dict(zip(('near','mid','far','standard'),[round(x) for x in buckets.cpu().tolist()]))
                    standard=bucket_worlds['standard']
                    cloud.update(match_metrics(name,values,reward_values[index],used_amplitude,
                        group.match_task.control,standard,bucket_worlds['near'],bucket_worlds))
                    group.course_amplitude=group.match_task.control.amplitude
                    sl=env.groups[name][0] if parallel else slice(None)
                    rollout.obs[-1,sl,:,-1]=group.course_amplitude
                    if parallel:env._obs[sl,:,-1]=group.course_amplitude
            update_match_controls()
            cloud['debug/比赛任务修订']=MATCH_TASK_REVISION
            cloud['debug/比赛步态奖励启用']=int(options.match_gait_reward)
            cloud['debug/运动约束启用']=int(motion_config['enabled'])
            cloud['debug/运动约束过渡比例']=min(1.,max(0.,(iteration-motion_config['start_iteration'])/64)) if motion_config['enabled'] else 0.
            cloud['debug/history_frames']=args.history_frames
            cloud['debug/amp_coefficient']=options.amp_coef
            if amp is not None:
                cloud['debug/amp_revision']=AMP_REVISION
                cloud['debug/amp_advantage_weight']=args.amp_advantage_weight
            for key in ('amp_loss','amp_gradient_penalty','amp_expert_score','amp_policy_score',
                        'amp_eligible_fraction','amp_effective_policy_score','amp_effective_style','amp_normalizer_samples'):
                if key in metrics:cloud['debug/'+key]=metrics[key]
            if args.amp_groups:
                for side in (('shared',) if shared else ('red','blue')):
                    for key in ('task_advantage_abs','amp_advantage_abs','task_value_loss','amp_value_loss'):
                        cloud['debug/'+side+'_'+key]=metrics[side+'/'+key]
            if history is not None:
                cloud['debug/history_output_weight_norm']=teams[0][0].actor.base.history_encoder.layers[-1].weight.norm().item()
            if not parallel and env.course in FULL_MATCHES:cloud['01_总览/回合摔倒率']=cm['fall_rate']
            for rank,memory in enumerate(memories):
                cloud[f'04_性能/GPU{rank}显存_GiB']=memory
            for key,label in [('slip_mps','支撑脚滑移_mps'),('phase_match','接触相位匹配率'),('velocity_error','速度误差_mps'),
                              ('left_steps','左脚有效步数'),('right_steps','右脚有效步数'),('clearance_m','足底离地_米'),
                              ('weight_shifts','有效重心切换'),('nonfoot_contact','非脚部着地比例'),
                              ('alternations','有效交替落脚'),('phase_lift','相位抬脚得分')]:
                cloud['05_步态/'+label]=None if key in ('phase_match','velocity_error') and not (any(s in GAIT_COURSES for s in env.allocations) if parallel else env.course in GAIT_COURSES) else gm[key]
            for side in (('shared',) if shared else ('red','blue')):
                for k in ('value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio','approx_kl','clip_fraction'):
                    cloud['debug/'+side+'_'+k] = metrics[side+'/'+k]
            if distributed.primary:
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(metrics,telemetry=cloud))+'\n')
            for k, v in cloud.items():
                if writer and isinstance(v, (float, int)):
                    writer.add_scalar(k, v, iteration)
            if writer:writer.flush()
            if wandb_run:
                cloud[COURSE_TABLE_KEY]=wandb.Html(course_table_html(cloud,iteration,parallel),inject=False,data_is_not_path=True)
                wandb_run.log(cloud, step=iteration)
            if distributed.primary:print(json.dumps(metrics), flush=True)
            stop = bool(distributed.maximum(int(stop)))
            old = curriculum.state_dict()
            if curriculum.advance():
                if parallel:
                    for name in COURSES:
                        if name in curriculum.mastered-set(old['mastered']):save(iteration,'stage_'+name+'.pt')
                else:
                    promoted = curriculum.state_dict()
                    curriculum.load_state_dict(old)
                    save(iteration, 'stage_'+curriculum.stage+'.pt')
                    curriculum.load_state_dict(promoted)
                # Commit the new gates and completed update before risky scene rebuilding.
                save(iteration)
                reset_stage()
            else:
                rollout.obs[0].copy_(rollout.obs[-1])
                if history is not None:rollout.history[0].copy_(rollout.history[-1])
            if iteration % options.save_interval == 0 or stop:
                save(iteration)
            if stop:
                break
        save(iteration)
    except BaseException as error:
        exit_code=1
        diagnostic={'error':str(error),'iteration':iteration,'last_completed_iteration':last_saved,'course':env.course}
        try:
            if parallel:diagnostic['groups']=env.diagnostics()
            else:
                import warp as wp
                diagnostic['latched_flags']=wp.to_torch(env.failure).cpu().tolist()
                diagnostic['overflow_flags']=wp.to_torch(env.data.overflow).cpu().tolist()
        except Exception as secondary: diagnostic['diagnostic_error']=str(secondary)
        failure_name = 'physics_failure.json' if distributed.primary else f'physics_failure_rank{distributed.rank}.json'
        (out/failure_name).write_text(json.dumps(diagnostic,indent=2))
        raise
    finally:
        distributed.close()
        if writer:writer.close()
        if wandb_run:
            wandb_run.finish(exit_code=exit_code)
    return out/'last.pt'
