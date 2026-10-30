"""Experimental SuperMimic-style Q1 visual soccer downstream integration.

Uses a motor initialization and a separate goal encoder with configurable
vision/truth input and frozen/joint decoder updates. Autoregressive dynamics
provide imagined-return gradients. Not a certified motor
skill library or a claim of kicking success. No archived soccer/CSV imports.
"""
import argparse
import copy
import json
import hashlib
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from q1_supermimic.soccer import load_config,SoccerEnvironment,Curriculum,PHASES
from q1_supermimic.networks import MotorPolicy,mlp
from q1_supermimic.soccer_dynamics import SoccerDynamics
from q1_supermimic.replay import Replay
from q1_supermimic.soccer_learning import soccer_update


def make_policy(model,observe,motor,width,visual_history=50,train_decoder=False):
    policy=MotorPolicy(720,540,7*visual_history,22,motor_widths=(width,width),visual_widths=(256,256))
    source=motor.get('accepted_policy',motor['policy'])
    policy.load_state_dict({key:value for key,value in source.items() if not key.startswith('visual_encoder.')},strict=False)
    # Start at exactly the reference-interface motor action, not a random
    # visual latent. The final goal layer learns immediately through decoder.
    with torch.no_grad():
        ref=torch.tensor(np.r_[model.qpos0,np.zeros(model.nv)],dtype=torch.float32).unsqueeze(0)
        ref[:,observe.control.qpos_indices.cpu()]=observe.control.neutral.cpu()
        goal=copy.deepcopy(observe).cpu().reference(ref).repeat(1,10)
        raw_latent=policy.reference_encoder(goal)[0]
        torch.nn.init.zeros_(policy.visual_encoder[-1].weight)
        policy.visual_encoder[-1].bias.copy_(raw_latent)
    policy.decoder.requires_grad_(train_decoder)
    policy.reference_encoder.requires_grad_(False)
    return policy


@torch.no_grad()
def collect(env,policy,replay,steps,noise):
    for _ in range(steps):
        action=policy.soccer(env.history.flatten(1),env.visual.flatten(1))
        if noise:
            action+=noise*torch.randn(action.shape,device=env.device,generator=env.generator)
        transition=env.step(action)
        if replay is not None:replay.append(**transition)


@torch.no_grad()
def evaluate(env,policy,phase):
    """One complete first episode per independent world; no reset-frequency bias."""
    env.phase=phase
    env.tracker.records.clear();env.tracker.completed=0
    env.reset(torch.arange(env.count,device=env.device))
    first={}
    for _ in range(env.config['episode_steps']):
        action=policy.soccer(env.history.flatten(1),env.visual.flatten(1))
        transition=env.step(action)
        records=list(env.tracker.records)
        # Records carry episode world id, so only first complete episodes count.
        for record in records:
            if record['world'] not in first:
                # Exact mesh/sphere gap at the terminal state, before reset.
                import mujoco
                data=mujoco.MjData(env.model)
                data.qpos[:]=transition['next_state'][record['world'],:env.model.nq].cpu().numpy()
                mujoco.mj_forward(env.model,data)
                record['final_foot_gap_m']=env.head.geometry.native_gap(data)
                record['foot_gap_reduction_m']=record['initial_foot_gap_m']-record['final_foot_gap_m']
                first[record['world']]=record
        env.tracker.records.clear()
        if len(first)==env.count:break
    records=list(first.values())
    if len(records)!=env.count:
        raise RuntimeError('Evaluation did not complete one episode per world')
    summary=dict(completed_episodes=len(records),
        effective_kick_success_rate=float(np.mean([r['effective_kick'] for r in records])),
        ball_goal_net_progress_m=float(np.mean([r['progress'] for r in records])),
        full_episode_fall_rate=float(np.mean([r['fall'] for r in records])),
        contact_rate=float(np.mean([r['contact'] for r in records])),
        goal_rate=float(np.mean([r['goal'] for r in records])),
        near_foot_gap_reduction_m=float(np.mean([r['foot_gap_reduction_m'] for r in records if r['group']==2])),
        approach_progress_m=float(np.mean([r['approach_progress_m'] for r in records])))
    return summary,records


@torch.no_grad()
def contact_cohort(config,policy,seed):
    cohort=SoccerEnvironment(config['contact_worlds'],config,seed)
    state=cohort.state()
    g=cohort.head.geometry
    relative=state[:,:2]-state[:,g.ball_q:g.ball_q+2]
    direction=relative/relative.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    near=cohort.groups==2
    state[near,cohort.model.nq+g.ball_v:cohort.model.nq+g.ball_v+2]=direction[near]*3.
    ids=torch.arange(cohort.count,device=cohort.device)
    cohort.physics.reset_indices(ids,state[:,:cohort.model.nq],state[:,cohort.model.nq:])
    replay=Replay(config['contact_steps'])
    collect(cohort,policy,replay,config['contact_steps'],0.)
    # Never use these privileged/excited episodes for task success or promotion.
    return replay,int(replay.storage['contact'][:replay.size].sum())


@torch.no_grad()
def contact_validation(world,batch,geometry):
    state=batch['state'][0]
    position_errors,velocity_errors,persistence_velocities,events=[],[],[],[]
    initial=batch['state'][0]
    for index in range(len(batch['action'])):
        state=world(state,batch['action'][index],batch['context'][index])
        target=batch['next_state'][index]
        bq,bv=geometry.ball_q,geometry.nq+geometry.ball_v
        position_errors.append((state[:,bq:bq+3]-target[:,bq:bq+3]).square().sum(-1))
        velocity_errors.append((state[:,bv:bv+3]-target[:,bv:bv+3]).square().sum(-1))
        persistence_velocities.append((initial[:,bv:bv+3]-target[:,bv:bv+3]).square().sum(-1))
        events.append(batch['contact'][index])
    mask=torch.stack(events)
    count=int(mask.sum())
    result=dict(sampled_contact_predictions=count,validated=False)
    if count:
        result.update(ball_position_rmse_m=float(torch.stack(position_errors)[mask].mean().sqrt()),
                      ball_velocity_rmse_mps=float(torch.stack(velocity_errors)[mask].mean().sqrt()),
                      persistence_ball_velocity_rmse_mps=float(torch.stack(persistence_velocities)[mask].mean().sqrt()))
    return result


def select_candidate(candidate,baseline,phase,pending,minimum_gap_gain=.001):
    """Real-physics retention; approach shaping does not unlock curriculum."""
    if not pending:return False,'no_actor_update'
    if not all(np.isfinite(list(metrics.values())).all() for metrics in (candidate,baseline)):
        return False,'nonfinite_evaluation'
    if candidate['full_episode_fall_rate']>baseline['full_episode_fall_rate']:
        return False,'fall_regression'
    primary='goal_rate' if phase==3 else 'effective_kick_success_rate'
    if candidate[primary]<baseline[primary]:return False,'task_regression'
    keys=[primary]+(['contact_rate'] if phase==0 else [])+['ball_goal_net_progress_m']
    score=tuple(candidate[key] for key in keys)
    old=tuple(baseline[key] for key in keys)
    if score>old:return True,'task_progress'
    if phase==0 and score==old and (candidate['near_foot_gap_reduction_m']-
            baseline['near_foot_gap_reduction_m']>=minimum_gap_gain):
        return True,'approach_shaping'
    return False,'no_real_progress'


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motor-checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--run-name')
    p.add_argument('--smoke',action='store_true',help='Bounded 2-update integration check, not performance acceptance')
    for field in ('num-envs','iterations','warmup-batches','evaluation-interval','evaluation-episodes','episode-steps'):
        p.add_argument('--'+field,type=int)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text()) if args.config else load_config()
    for key in ('num_envs','iterations','warmup_batches','evaluation_interval','evaluation_episodes','episode_steps'):
        if getattr(args,key) is not None:cfg[key]=getattr(args,key)
    if args.smoke:
        cfg.update(num_envs=8,iterations=2,collect_steps=16,warmup_steps=32,warmup_batches=10,
                   model_batches=2,model_batch_size=32,policy_batch_size=4,model_width=64,
                   imagination_horizon=4,evaluation_interval=2,evaluation_episodes=8,episode_steps=64)
    if cfg['num_envs']%4 or cfg['evaluation_episodes']%4:
        p.error('Environment/evaluation counts must be multiples of four')
    if any(not isinstance(cfg[key],int) or cfg[key]<1 for key in (
        'num_envs','iterations','collect_steps','replay_steps','warmup_steps','warmup_batches','model_batches',
        'model_batch_size','policy_batch_size','model_width','imagination_horizon','evaluation_interval',
        'evaluation_episodes','episode_steps','save_interval','threads','max_stalled_evaluations',
        'visual_history','contact_worlds','contact_steps')):
        p.error('Counts must be positive integers')
    if cfg['warmup_steps']<8 or cfg['replay_steps']<8:
        p.error('Need at least 8 real steps for dynamics windows')
    if cfg['spawn_mixture']!=[.25,.25,.5] or cfg['near_foot_gap']!=[.1,.25] or cfg['spawn_disk_radius']!=3.5:
        p.error('Fixed spawn contract cannot be changed')
    if cfg.get('goal_input_mode','vision') not in ('vision','truth'):
        p.error('Unknown goal input mode')
    if not isinstance(cfg.get('train_decoder',False),bool):p.error('train_decoder must be boolean')
    return args,cfg


def main():
    args,cfg=parse()
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True,exist_ok=True)
    motor=torch.load(args.motor_checkpoint,map_location='cpu',weights_only=False)
    width=motor.get('config',{}).get('policy_width',128)
    env=SoccerEnvironment(cfg['num_envs'],cfg,cfg['seed'])
    policy=make_policy(env.model,env.observe,motor,width,cfg['visual_history'],cfg.get('train_decoder',False)).to(env.device)
    motor_iteration=motor['iteration']
    digest=hashlib.sha256()
    for key,value in policy.decoder.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    motor_decoder_hash=digest.hexdigest()
    torch.save(dict(iteration=motor_iteration,decoder=policy.decoder.state_dict(),reference_encoder=policy.reference_encoder.state_dict(),
                    deployment_accepted=False,sha256=motor_decoder_hash),args.output/'motor_snapshot.pt')
    del motor
    world=SoccerDynamics(env.model,cfg,widths=(cfg['model_width'],)*3).to(env.device)
    critic=mlp(env.model.nq+env.model.nv+7*cfg['visual_history']+4,1,(512,256)).to(env.device)
    actor_groups=[dict(params=list(policy.visual_encoder.parameters()),lr=cfg['actor_lr'])]
    if cfg.get('train_decoder',False):
        actor_groups.append(dict(params=list(policy.decoder.parameters()),lr=cfg.get('decoder_lr',cfg['actor_lr'])))
    actor_opt=torch.optim.Adam(actor_groups)
    model_opt=torch.optim.Adam(world.parameters(),lr=cfg['learning_rate'])
    critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg['learning_rate'])
    replay=Replay(cfg['replay_steps'])
    curriculum=Curriculum(cfg)
    baseline_policy=copy.deepcopy(policy)
    accepted_optimizer=copy.deepcopy(actor_opt.state_dict())
    start=0;saved=None
    if args.resume:
        saved=torch.load(args.resume,map_location=env.device,weights_only=False)
        if saved['config']!=cfg:raise ValueError('Resume must preserve exact soccer config')
        for key,module in [('policy',policy),('world',world),('critic',critic),('accepted_policy',baseline_policy)]:
            module.load_state_dict(saved[key])
        for key,optimizer in [('actor_opt',actor_opt),('model_opt',model_opt),('critic_opt',critic_opt)]:
            optimizer.load_state_dict(saved[key])
        accepted_optimizer=saved['accepted_actor_optimizer']
        replay.storage,replay.size,replay.cursor=saved['replay'],saved['replay_size'],saved['replay_cursor']
        curriculum.phase=saved['phase'];env.phase=curriculum.phase;start=saved['iteration']
        env.generator.set_state(saved['collection_rng'].cpu())
        env.sampler.random.bit_generator.state=saved['spawn_rng']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    run=None
    scope=('experimental privileged soccer with jointly trained motor decoder' if cfg.get('train_decoder',False)
           else 'experimental visual soccer downstream; frozen unvalidated standing motor decoder')
    config=dict(cfg,scope=scope,
                motor_checkpoint=str(args.motor_checkpoint.resolve()),policy_width=width,
                motor_source_iteration=motor_iteration,motor_decoder_sha256=motor_decoder_hash,
                deployment_accepted=False,contact_model_validated=False,smoke=args.smoke)
    if args.wandb:
        import wandb
        run=wandb.init(project=cfg['wandb_project'],entity=cfg['wandb_entity'],dir=str(args.output),
             config=config,name=args.run_name,job_type='soccer-truth-joint' if cfg.get('train_decoder',False) else 'soccer-downstream',mode='online',
             id=saved.get('wandb_run_id') if saved else None,
             resume='must' if saved and saved.get('wandb_run_id') else None)
        config['wandb_url']=run.url
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def log(values,step):
        print(json.dumps(values),flush=True)
        with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(values)+'\n')
        if run:
            flattened={}
            for key,value in values.items():
                if isinstance(value,dict):
                    flattened.update({key+'/'+k:v for k,v in value.items() if isinstance(v,(int,float,bool))})
                elif isinstance(value,(int,float,bool,str)):flattened[key]=value
            run.log(flattened,step=step)
    # Independent validation worlds, seeds and episodes, no adjacent-frame split.
    validation_env=SoccerEnvironment(min(32,cfg['num_envs']),cfg,194947)
    validation=Replay(32)
    collect(validation_env,baseline_policy,validation,32,cfg['exploration_std'])
    held_out=validation.sample(cfg['model_batch_size'])
    del validation,validation_env
    contact_train,contact_events=contact_cohort(cfg,baseline_policy,582991)
    contact_test,test_contact_events=contact_cohort(cfg,baseline_policy,982993)
    if not contact_events or not test_contact_events:
        raise RuntimeError('Contact excitation did not produce real foot/ball contacts')
    contact_held_out=contact_test.sample(cfg['model_batch_size'],require_contact=True)
    del contact_test
    print(json.dumps(dict(contact_training_transitions=contact_events,contact_validation_transitions=test_contact_events)),flush=True)
    def model_batch():
        count=cfg['model_batch_size']//2
        task=replay.sample(count)
        contact=contact_train.sample(cfg['model_batch_size']-count,require_contact=True)
        return {key:torch.cat((task[key],contact[key]),1) for key in task}
    # Fresh real trajectories after resume are intentional. No claim of bitwise
    # continuation across reconstructed simulator states or episode queues.
    collect(env,baseline_policy,replay,cfg['warmup_steps'],cfg['exploration_std'])
    if not saved:
        world.set_scales(replay.storage['state'][:replay.size])
        bq,bv=env.head.geometry.ball_q,env.head.geometry.ball_v
        world.scale[bq:bq+3].clamp_min_(1.)
        world.scale[env.model.nq+bv:env.model.nq+bv+3].clamp_min_(1.)
        world.scale[env.model.nq+bv+3:env.model.nq+bv+6].clamp_min_(10.)
        world.output_scale[bv:bv+3].clamp_min_(.002)
        world.output_scale[env.model.nv+bv:env.model.nv+bv+3].clamp_min_(.1)
        world.output_scale[env.model.nv+bv+3:env.model.nv+bv+6].clamp_min_(1.)
        for index in range(cfg['warmup_batches']):
            model_opt.zero_grad(set_to_none=True)
            loss=world.window_loss(model_batch())
            loss.backward();torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True);model_opt.step()
            if (index+1)%50==0:print(json.dumps(dict(warmup_batch=index+1,model_loss=float(loss.detach()))),flush=True)
    evaluation_env=SoccerEnvironment(cfg['evaluation_episodes'],cfg,324161)
    # Preserve paired reset seeds for candidate/incumbent comparisons.
    def evaluate_paired(candidate):
        evaluation_env.sampler.random=np.random.default_rng(324161)
        evaluation_env.generator.manual_seed(324162)
        return evaluate(evaluation_env,candidate,curriculum.phase)
    baseline,baseline_records=evaluate_paired(baseline_policy)
    log(dict(iteration=start,phase=PHASES[curriculum.phase],evaluation=baseline),start)
    iteration=start
    last_progress=saved.get('last_progress',start) if saved else start
    pending=saved.get('pending_actor_updates',0) if saved else 0
    def checkpoint():
        saved=dict(iteration=iteration,config=cfg,phase=curriculum.phase,last_progress=last_progress,
           pending_actor_updates=pending,
           policy=policy.state_dict(),world=world.state_dict(),critic=critic.state_dict(),
           accepted_policy=baseline_policy.state_dict(),accepted_actor_optimizer=accepted_optimizer,
           actor_opt=actor_opt.state_dict(),model_opt=model_opt.state_dict(),critic_opt=critic_opt.state_dict(),
           replay=replay.storage,replay_size=replay.size,replay_cursor=replay.cursor,
           collection_rng=env.generator.get_state(),spawn_rng=env.sampler.random.bit_generator.state,
           torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
           wandb_run_id=run.id if run else None,evaluation=baseline,deployment_accepted=False,
           scope=config['scope'],motor_checkpoint=config['motor_checkpoint'])
        temporary=args.output/'checkpoint.tmp';torch.save(saved,temporary);temporary.replace(args.output/'last.pt')
    checkpoint()
    reason='iterations_complete'
    for iteration in range(start+1,start+cfg['iterations']+1):
        tick=time.perf_counter()
        collect(env,baseline_policy,replay,cfg['collect_steps'],cfg['exploration_std'])
        for _ in range(cfg['model_batches']):
            model_opt.zero_grad(set_to_none=True)
            loss=world.window_loss(model_batch())
            loss.backward();torch.nn.utils.clip_grad_norm_(world.parameters(),1.,error_if_nonfinite=True);model_opt.step()
        with torch.no_grad():
            validation_loss=float(world.window_loss(held_out))
            persistence=float(world.error(held_out['state'][0].expand_as(held_out['next_state']),held_out['next_state']))
        contact_metrics=contact_validation(world,contact_held_out,env.head.geometry)
        contact_ready=bool(contact_metrics.get('ball_velocity_rmse_mps',float('inf'))<
                           contact_metrics.get('persistence_ball_velocity_rmse_mps',0.))
        model_ready=bool(np.isfinite(validation_loss) and validation_loss<persistence and
                         (not cfg.get('train_decoder',False) or contact_ready))
        result=dict(iteration=iteration,phase=PHASES[curriculum.phase],model_loss=float(loss.detach()),
           validation=dict(model_loss=validation_loss,persistence_loss=persistence,beats_persistence=bool(model_ready)),
           training=env.tracker.summary(),contact_model=contact_metrics,
           goal_input_mode=cfg.get('goal_input_mode','vision'),train_decoder=cfg.get('train_decoder',False),
           observation=dict(ball_available_fraction=float(env.visual[:,-1,2].mean()),
                 near_ball_available_fraction=float(env.visual[env.groups==2,-1,2].mean())))
        # Smoke deliberately exercises connectivity before a trained model exists.
        # It is never used as an acceptance result.
        if model_ready or args.smoke:
            batch=replay.sample(cfg['policy_batch_size'],length=1)
            result.update(soccer_update(policy,world,critic,env.head,env.observe,env.camera,
                  actor_opt,critic_opt,batch,phase=curriculum.phase,horizon=cfg['imagination_horizon'],generator=env.generator,train_decoder=cfg.get('train_decoder',False)))
            pending+=1
        result['actor_update_applied']=bool(model_ready or args.smoke)
        if iteration%cfg['evaluation_interval']==0 or iteration==start+cfg['iterations'] or stop:
            candidate,records=evaluate_paired(policy)
            # Real task metrics decide retention, never imagined reward alone.
            retained,selection_reason=select_candidate(candidate,baseline,curriculum.phase,pending,
                cfg.get('minimum_approach_selection_gain_m',.001))
            result['evaluation']=candidate
            result['selection']=dict(retained=retained,pending_actor_updates=pending,reason=selection_reason)
            if retained:
                baseline_policy.load_state_dict(policy.state_dict());accepted_optimizer=copy.deepcopy(actor_opt.state_dict())
                baseline,baseline_records=candidate,records;last_progress=iteration
            else:
                policy.load_state_dict(baseline_policy.state_dict());actor_opt.load_state_dict(accepted_optimizer)
            pending=0
            if curriculum.consider(baseline_records):
                env.phase=curriculum.phase
                baseline,baseline_records=evaluate_paired(baseline_policy)
                result['promoted_phase']=PHASES[curriculum.phase]
            (args.output/'evaluation.json').write_text(json.dumps(dict(iteration=iteration,metrics=candidate,episodes=records),indent=2)+'\n')
        torch.cuda.synchronize()
        result['performance']=dict(iteration_seconds=time.perf_counter()-tick,
             gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,gpu_reserved_mib=torch.cuda.memory_reserved()/2**20)
        result['iterations_since_retained_update']=iteration-last_progress
        log(result,iteration)
        stalled=iteration-last_progress>=cfg['evaluation_interval']*cfg['max_stalled_evaluations']
        if iteration%cfg['save_interval']==0 or iteration==start+cfg['iterations'] or stop or stalled:checkpoint()
        if stop:reason='signal';break
        if stalled:reason='no_retained_policy_progress';break
    (args.output/'completion.json').write_text(json.dumps(dict(iteration=iteration,reason=reason,
          phase=PHASES[curriculum.phase],evaluation=baseline,deployment_accepted=False),indent=2)+'\n')
    if run:run.summary.update(dict(completion_reason=reason,deployment_accepted=False));run.finish()


if __name__=='__main__':main()
