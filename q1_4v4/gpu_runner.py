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
from .learning import make_args, make_team, save_team, load_team, policy_sharing, with_team_identity
from .course_config import Curriculum, LOW_LEVEL, DURATIONS, COURSES, GAIT_COURSES, BALL_TASK_REVISION, CONTESTS
from .distributed import Distributed
from .gait import METRIC_NAMES, REWARD_REVISION
from .parallel_curriculum import ParallelCurriculum
from .mixed_env import MixedCourseEnv, ended_rows
from .telemetry import SKILL_NAMES, active_team_reward, course_metrics, macro_metrics
from .course_table import COURSE_TABLE_KEY,course_table_html


def run(options):
    if min(options.num_envs, options.horizon, options.epochs, options.iterations, options.save_interval) < 1:
        raise ValueError('Training sizes must be positive')
    if options.episode_steps is not None and options.episode_steps < 1:
        raise ValueError('episode-steps must be positive')
    device = torch.device(options.device)
    if device.type != 'cuda':
        raise ValueError('--backend warp requires --device cuda:0')
    distributed = Distributed(device)
    torch.set_num_threads(2)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    args = make_args(options.num_envs, options.horizon, options.epochs)
    if options.learning_rate is not None:
        if options.learning_rate<=0:raise ValueError('learning-rate must be positive')
        args.lr=args.critic_lr=options.learning_rate
    checkpoint = torch.load(options.checkpoint,map_location=device,weights_only=False) if options.checkpoint else None
    options.curriculum_mode = getattr(options,'curriculum_mode',None) or (checkpoint.get('curriculum_mode','serial') if checkpoint else 'serial')
    parallel = options.curriculum_mode=='parallel'
    curriculum = ParallelCurriculum() if parallel else Curriculum(options.stage)
    start = 0
    if checkpoint:
        curriculum.load_state_dict(checkpoint['curriculum'])
        start = checkpoint['iteration']
    if options.start_course is not None:
        if parallel:raise ValueError('--start-course is only for serial curricula')
        if not options.checkpoint: raise ValueError('--start-course requires --checkpoint')
        selected = COURSES[int(options.start_course)] if options.start_course.isdigit() else options.start_course
        if selected not in COURSES: raise ValueError('Unknown start course')
        curriculum=Curriculum(selected)
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
                             nconmax=options.nconmax,njmax=options.njmax,episode_steps=options.episode_steps)
    else:
        env = CourseWarpEnv(options.num_envs, str(device), options.seed, curriculum.stage,
                            nconmax=options.nconmax, njmax=options.njmax)
    teams = [make_team(args, env, device, gpu_rollout=True,shared=shared) for _ in groups]
    if checkpoint:
        if checkpoint.get('backend') != 'warp':
            raise ValueError('Use a Warp checkpoint for continuation; CPU legacy runs are reference-only')
        if len(checkpoint['teams']) != len(teams):
            raise ValueError('Checkpoint policy count does not match policy sharing')
        for state, (policy, trainer, _) in zip(checkpoint['teams'], teams):
            load_team(state, policy, trainer)
            if checkpoint.get('reward_version',1) != REWARD_REVISION:
                trainer.value_normalizer.running_mean.zero_()
                trainer.value_normalizer.running_mean_sq.zero_()
                trainer.value_normalizer.debiasing_term.zero_()
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
    rollout = GpuRollout(options.horizon, options.num_envs, device, obs_dim=raw_obs_dim)
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
        config = dict(vars(options), world_size=distributed.world_size, per_course_metrics=True, ball_task_revision=BALL_TASK_REVISION,
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
        obs, _ = env.reset()
        rollout.obs[0].copy_(obs)
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
        if name=='last.pt':last_saved=iteration
        if not distributed.primary:
            return
        payload = dict(version=4, reward_version=REWARD_REVISION, ball_task_revision=BALL_TASK_REVISION, obs_dim=obs_dim, raw_obs_dim=raw_obs_dim,
                       policy_sharing=options.policy_sharing, curriculum_mode=options.curriculum_mode, backend='warp', iteration=iteration, args=vars(args),
                       teams=[save_team(p, t) for p, t, _ in teams], curriculum=curriculum.state_dict(),
                       **rng, distributed_rng=rank_rng, world_size=distributed.world_size,
                       parameter_max_difference=difference)
        temp = out/(name+'.tmp')
        torch.save(payload, temp)
        temp.replace(out/name)

    reset_stage()
    iteration = start
    try:
        if options.wandb and distributed.primary:
            import wandb
            wandb_run = wandb.init(entity='mohismlab-robotics', project='q1-humanoid-kick-4v4',
                name=options.run_name or out.name, job_type=options.job_type,
                tags=['q1', '4v4', 'mappo', 'warp', options.job_type], config=vars(options),
                dir=str(out.resolve()), settings=wandb.Settings(init_timeout=60))
            wandb_run.config.update(dict(vars(options), world_size=distributed.world_size,
                total_num_envs=options.num_envs*distributed.world_size,
                reward_revision=REWARD_REVISION, curriculum_window_policy='per_team', per_course_metrics=True, ball_task_revision=BALL_TASK_REVISION), allow_val_change=True)
            wandb_run.define_metric('iteration')
            wandb_run.define_metric('*', step_metric='iteration')
            wandb_run.define_metric('debug/*', hidden=True)
            (out/'wandb_run.json').write_text(json.dumps({'id': wandb_run.id, 'url': wandb_run.url}))
        distributed.maximum(0)  # All workers wait for primary W&B initialization.
        for iteration in range(start+1, start+options.iterations+1):
            torch.cuda.synchronize(device)
            tic = time.perf_counter()
            completed = []
            raw_reward = torch.zeros(2, device=device)
            skill_stats = torch.zeros(len(SKILL_NAMES),device=device)
            gait_stats = torch.zeros(len(METRIC_NAMES),device=device)
            if parallel:env.begin_rollout()
            for step in range(options.horizon):
                rollout.active[step].copy_(env.active)
                obs = rollout.obs[step]
                state = env.state(obs)
                with torch.no_grad():
                    actor_obs,critic_state=policy_inputs(obs,state)
                    for t, (policy, trainer, _) in enumerate(teams):
                        sl = groups[t]
                        trainer.prep_rollout()
                        value, action, logprob, _, _ = policy.get_actions(critic_state[:, sl].reshape(-1, state_dim),
                            actor_obs[:, sl].reshape(-1, obs_dim), rnn, rnn, ones)
                        rollout.actions[step, :, sl].copy_(action.reshape(options.num_envs, agents, 22))
                        rollout.log_probs[step, :, sl].copy_(logprob.reshape(options.num_envs, agents, 1))
                        rollout.values[step, :, sl].copy_(trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1))
                    terminal_obs, terminal_state, reward, terminated, truncated, info = env.step(rollout.actions[step])
                    gait_stats += torch.stack([env.gait_metrics[k] for k in METRIC_NAMES])
                    skill_stats += torch.stack([env.last_stats[k] for k in SKILL_NAMES])
                    raw_reward += active_team_reward(reward,env.active)
                    # Bootstrap timeout final states BEFORE resetting. No CPU value normalization.
                    _,terminal_state=policy_inputs(terminal_obs,terminal_state)
                    for t, (policy, trainer, _) in enumerate(teams):
                        sl = groups[t]
                        value = policy.get_values(terminal_state[:, sl].reshape(-1, state_dim), rnn, ones)
                        timeout_mask = truncated[:,sl,None] if truncated.ndim==2 else truncated[:,None,None]
                        reward[:, sl] += args.gamma*trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1)*timeout_mask
                    rollout.rewards[step].copy_(reward)
                    done = terminated | truncated
                    terminal_mask = done if done.ndim==2 else done[:,None].expand(-1,8)
                    rollout.masks[step].copy_((~terminal_mask)[:,:,None])
                    if parallel:
                        terminal_obs,ended=env.finish_step()
                        if len(ended):completed.append(ended)
                    else:
                        ended,ids=ended_rows(env,terminated,truncated,info)
                        if len(ended):
                            terminal_obs,_=env.reset_agents(ids) if env.course in LOW_LEVEL else env.reset(ids[:,0])
                            completed.append(ended)
                    rollout.obs[step+1].copy_(terminal_obs)
            env.check()  # Includes overflow accumulated during all substeps, before any optimizer update.
            with torch.no_grad():
                state = env.state(rollout.obs[-1])
                _,state=policy_inputs(rollout.obs[-1],state)
                for t, (policy, trainer, _) in enumerate(teams):
                    sl = groups[t]
                    value = policy.get_values(state[:, sl].reshape(-1, state_dim), rnn, ones)
                    rollout.values[-1, :, sl].copy_(trainer.value_normalizer.denormalize(value).reshape(options.num_envs, agents, 1))
                rollout.compute_returns(args.gamma, args.gae_lambda)
            metrics = {}
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
            if iteration % options.save_interval == 0 or stop:
                save(iteration)
            if stop:
                break
        save(iteration)
    except BaseException as error:
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
        if wandb_run:
            wandb_run.finish(exit_code=1)
            wandb_run=None
        raise
    finally:
        if writer:writer.close()
        if wandb_run:
            wandb_run.finish()
    if distributed.world_size > 1:
        torch.distributed.destroy_process_group()
    return out/'last.pt'
