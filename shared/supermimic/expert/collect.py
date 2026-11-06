"""Collect complete native-MuJoCo expert episodes, discarding unsuccessful runs.

This runs a frozen teacher only. No SuperMimic policy/world-model optimization.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import signal
import time

import mujoco
import numpy as np
import torch

from .runtime import DEFAULT_CHECKPOINT,ROOT,SoccerPlayer
from .dataset import SCHEMA_VERSION,observation_history,outcome,validate_arrays


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


class EpisodeBuilder:
    def __init__(self,player):
        self.player=player;self.substeps=player.decimation
        self.qpos=[player.data.qpos.copy()];self.qvel=[player.data.qvel.copy()]
        self.warmstart=[player.data.qacc_warmstart.copy()];self.times=[float(player.data.time)]
        self.actions=[];self.raw=[];self.targets=[];self.obs=[];self.queues=[]
        self.torques=[];self.contacts=[];self.other=[];self.progress=[];self.post_progress=[]
        self.pending=None;self.initial_history=None;self.last_history=None
        self.first_touch_time=None;self.goal_distance_at_touch=None
        self.start_goal_distance=self.goal_distance()
        self.previous_physics_time=float(player.data.time)
        foot_bodies={player.model.body(name).id for name in player.cfg['asset']['foot_names']}
        self.foot_geoms={g for g in range(player.model.ngeom) if player.model.geom_bodyid[g] in foot_bodies and player.model.geom_contype[g]}
        self.ball_geom=player.model.geom('soccer_ball').id

    def goal_distance(self):
        p=self.player
        return float(np.linalg.norm(p.data.qpos[p.ball_addr:p.ball_addr+2]-[7.,0.]))

    def on_control(self,obs,history,raw):
        p=self.player
        if self.pending is not None:raise RuntimeError('Control boundary crossed incomplete physics transition')
        if self.initial_history is None:self.initial_history=history.copy()
        else:
            expected=np.concatenate((obs[None],self.last_history[:-1]),axis=0)
            if not np.array_equal(history,expected):raise AssertionError('Teacher history changed order or reset mid-episode')
        self.last_history=history.copy()
        self.obs.append(obs);self.actions.append(p.actions.copy());self.raw.append(raw)
        self.targets.append(p.targets.copy());self.queues.append(p.ball_history.copy())
        self.pending=dict(torques=[],contact=[],other=[])

    def on_physics(self):
        p=self.player;pending=self.pending
        if pending is None:raise RuntimeError('Physics without executed control')
        if p.data.time<=self.previous_physics_time:
            raise FloatingPointError('Native simulator reset time inside a transition')
        self.previous_physics_time=float(p.data.time)
        if not all(np.isfinite(value).all() for value in [p.data.qpos,p.data.qvel,p.data.ctrl,p.data.qacc_warmstart]):
            raise FloatingPointError('Non-finite expert transition')
        foot=False;other=False
        for contact in p.data.contact:
            if contact.dist>0 or contact.efc_address<0 or self.ball_geom not in contact.geom:continue
            partner=int(contact.geom[0] if contact.geom[1]==self.ball_geom else contact.geom[1])
            if partner in self.foot_geoms:foot=True
            elif p.model.geom_bodyid[partner]!=0:other=True
        if foot and self.first_touch_time is None:
            self.first_touch_time=float(p.data.time)
            self.goal_distance_at_touch=self.goal_distance()
        pending['torques'].append(p.data.ctrl.copy());pending['contact'].append(foot);pending['other'].append(other)
        if len(pending['torques'])==self.substeps:
            self.torques.append(pending['torques']);self.contacts.append(pending['contact']);self.other.append(pending['other'])
            self.qpos.append(p.data.qpos.copy());self.qvel.append(p.data.qvel.copy())
            self.warmstart.append(p.data.qacc_warmstart.copy());self.times.append(float(p.data.time))
            distance=self.goal_distance();self.progress.append(self.start_goal_distance-distance)
            self.post_progress.append(0. if self.goal_distance_at_touch is None else self.goal_distance_at_touch-distance)
            self.pending=None

    @property
    def steps(self):return len(self.torques)

    def finish(self,cause):
        p=self.player;t=self.steps;duration=self.times[-1]-self.times[0]
        contacts=np.asarray(self.contacts,dtype=bool).reshape(t,self.substeps)
        # A nonfinite/incomplete final control is discarded; only finished,
        # finite physical transitions survive into the dataset.
        first=self.first_touch_time
        if not contacts.any():first=None
        post=max(self.post_progress,default=0.)
        labels=outcome(cause,contacts,post,duration,first)
        meta=dict(schema_version=SCHEMA_VERSION,nq=p.model.nq,nv=p.model.nv,
                  substeps=self.substeps,physics_dt=float(p.model.opt.timestep),
                  control_dt=float(p.model.opt.timestep*self.substeps),steps=t,duration_s=duration,
                  first_touch_s=first,post_touch_max_progress_m=max(0.,post),
                  ball_goal_net_progress_m=self.progress[-1] if t else 0.,
                  ball_delay_steps=p.ball_delay,spawn_group=p.spawn_kind,**labels)
        shape=lambda values,dtype,tail:np.asarray(values[:t],dtype=dtype).reshape((t,*tail))
        arrays=dict(qpos=np.asarray(self.qpos),qvel=np.asarray(self.qvel),
                    qacc_warmstart=np.asarray(self.warmstart),time=np.asarray(self.times),
                    actions=shape(self.actions,np.float32,(22,)),raw_actions=shape(self.raw,np.float32,(22,)),
                    pd_targets=shape(self.targets,np.float32,(22,)),observations=shape(self.obs,np.float32,(79,)),
                    initial_history=self.initial_history if self.initial_history is not None else np.zeros((50,79),dtype=np.float32),
                    ball_observation_queue=shape(self.queues,np.float32,(20,3)),
                    applied_torques=np.asarray(self.torques,dtype=np.float64).reshape(t,self.substeps,22),
                    foot_ball_contact=contacts,other_ball_contact=np.asarray(self.other,dtype=bool).reshape(t,self.substeps),
                    ball_goal_progress_m=np.asarray(self.progress),post_touch_goal_progress_m=np.asarray(self.post_progress),
                    terminated=np.zeros(t,dtype=bool),truncated=np.zeros(t,dtype=bool),episode_end=np.zeros(t,dtype=bool),
                    world_model_valid=np.ones(t,dtype=bool),policy_imitation_valid=np.full(t,labels['policy_imitation_eligible'],dtype=bool))
        if t:
            arrays['terminated'][-1]=labels['terminated'];arrays['truncated'][-1]=labels['truncated'];arrays['episode_end'][-1]=True
        validate_arrays(arrays,meta)
        return arrays,meta


def episode_seed(start,group):
    """Choose ordinary original-playback seeds to realize the fixed mixture."""
    seed=start
    while int(np.random.default_rng(seed).choice(3,p=[.25,.25,.5]))!=group:seed+=1
    return seed


def split_for(seed,checkpoint_hash):
    bucket=int(hashlib.sha256(f'{checkpoint_hash}:{seed}'.encode()).hexdigest()[:8],16)%10
    return 'test' if bucket<2 else 'validation' if bucket==2 else 'train'


def collect_episode(player,stop_requested=lambda:False,max_control_steps=None):
    builder=EpisodeBuilder(player);cause=None;error=None
    cap=max_control_steps or int(np.ceil(player.cfg['rewards']['episode_length_s']/(
                                player.model.opt.timestep*player.decimation)))+2
    try:
        while True:
            cause=player.step(auto_reset=False,on_control=builder.on_control,on_physics=builder.on_physics)
            if cause:break
            if builder.pending is None and (stop_requested() or builder.steps>=cap):
                # Exactly the next original terminal check; never check twice
                # and inadvertently shorten the goal's 50-control-step hold.
                cause=player._q1_episode_end() or ('interrupted' if stop_requested() else 'collector_limit')
                break
    except FloatingPointError as exc:
        cause='invalid';error=str(exc)
    arrays,meta=builder.finish(cause)
    if error:meta['error']=error
    return arrays,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=24)
    parser.add_argument('--seed',type=int,default=100000)
    parser.add_argument('--max-control-steps',type=int,help='Optional collection cutoff; never labelled timeout/success')
    parser.add_argument('--reset-mode',choices=['reference','default'],default='reference')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--bounded-targets',action='store_true',help='Clamp joint targets as well as final motor torques')
    args=parser.parse_args()
    if args.episodes<4 or args.episodes%4:parser.error('episodes must be a positive multiple of four')
    if args.max_control_steps is not None and args.max_control_steps<1:parser.error('max-control-steps must be positive')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Output must be empty; existing data is never overwritten')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    checkpoint_hash=digest(args.checkpoint)
    source_document=(ROOT/'SOURCE.json').read_bytes()
    (args.output/'source_bundle.json').write_bytes(source_document)
    (args.output/'config_snapshot.yaml').write_bytes((ROOT/'configs/Q1.yaml').read_bytes())
    implementation={name:digest(ROOT/name) for name in ['runtime.py','policy.py','motion.py','kinematics.py','torch_math.py','termination.py','collect.py','dataset.py']}
    manifest=dict(schema_version=SCHEMA_VERSION,created_at=datetime.datetime.now().astimezone().isoformat(),
       checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hash,source_bundle_sha256=hashlib.sha256(source_document).hexdigest(),
       config_sha256=digest(ROOT/'configs/Q1.yaml'),scene_sha256=digest(ROOT/'resources/q1_description/q1_soccer.xml'),
       implementation_sha256=implementation,planned_episodes=args.episodes,completed_episodes=0,
       reset_mode=args.reset_mode,spawn_mixture=[.25,.25,.5],history_order='newest_first',
       observation_dim=79,history_frames=50,action_dim=22,goal_input_mode='original_expert_vision',
       max_control_steps=args.max_control_steps,quaternion_order='wxyz',
       policy_inputs=['observations','causal 50-frame observation history'],
       privileged_fields=['qpos','qvel','qacc_warmstart','contact labels'],
       state_precision='float64',physics='native MuJoCo nominal',
       policy_imitation_rule='goal + real foot contact + >=0.1m post-contact goalward progress + >=1s recovery; no fall',
       world_model_rule='strict successful episodes only',selection_rule='strict_success_only',
       split_rule='sha256(checkpoint_hash:episode_seed), disjoint entire episodes; 70/10/20 train/validation/test',
       control_contract='joint_target_and_torque_limits_v1' if args.bounded_targets else 'original_expert',
       training_started=False,status='collecting')
    if args.bounded_targets:
        from .control import BoundedSoccerPlayer
        player_class=BoundedSoccerPlayer
        manifest['implementation_sha256']['control.py']=digest(ROOT/'control.py')
    else:
        player_class=SoccerPlayer
    (args.output/'index.jsonl').touch(exist_ok=False)
    atomic_json(args.output/'manifest.json',manifest)
    totals=Counter();seed_cursor=args.seed;started=time.perf_counter();summary_rows=[];attempts=Counter()
    for episode in range(args.episodes):
        if stop:break
        group=[0,1,2,2][episode%4];seed=episode_seed(seed_cursor,group);seed_cursor=seed+1
        player=player_class(args.checkpoint,seed=seed,reset_mode=args.reset_mode,task='Q1')
        if player.spawn_kind!=group:raise AssertionError('Original reset seed does not realize required cohort')
        initial_rng=json.loads(json.dumps(player.rng.bit_generator.state))
        arrays,meta=collect_episode(player,lambda:stop,args.max_control_steps)
        attempts[meta['termination']]+=1
        if not (meta['verified_success'] and meta['policy_imitation_eligible'] and meta['termination']=='goal'):
            manifest.update(attempted_episodes=sum(attempts.values()),discarded_attempts=sum(attempts.values())-len(summary_rows),attempt_outcomes=dict(attempts))
            atomic_json(args.output/'manifest.json',manifest)
            continue
        identifier=f'episode_{episode+1:06d}'
        meta.update(control_contract=manifest['control_contract'],episode_id=identifier,seed=seed,split=split_for(seed,checkpoint_hash),
                    checkpoint_sha256=checkpoint_hash,initial_sensor_rng=initial_rng,
                    final_sensor_rng=player.rng.bit_generator.state,reset_mode=args.reset_mode,
                    joint_names=[player.model.actuator(i).name for i in range(player.model.nu)],
                    initial_numpy_seed=seed,torch_seed=seed)
        temporary=args.output/(identifier+'.npz.tmp');path=args.output/(identifier+'.npz')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(path);meta['trajectory_sha256']=digest(path)
        atomic_json(args.output/(identifier+'.json'),meta)
        record={key:meta[key] for key in ['episode_id','seed','split','spawn_group','termination','task_success',
             'verified_success','policy_imitation_eligible','effective_kick','contact','timeout','failure','steps',
             'duration_s','ball_goal_net_progress_m','post_touch_max_progress_m','trajectory_sha256']}
        with (args.output/'index.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        summary_rows.append(record);totals[meta['termination']]+=1
        manifest.update(completed_episodes=len(summary_rows),outcomes=dict(totals))
        atomic_json(args.output/'manifest.json',manifest)
        print(json.dumps(record),flush=True)
    manifest.update(status='interrupted' if stop else 'complete',elapsed_seconds=time.perf_counter()-started,
           attempted_episodes=sum(attempts.values()),discarded_attempts=sum(attempts.values())-len(summary_rows),attempt_outcomes=dict(attempts),
           total_transitions=sum(r['steps'] for r in summary_rows),
           policy_imitation_episodes=sum(r['policy_imitation_eligible'] for r in summary_rows),
           splits=dict(Counter(r['split'] for r in summary_rows)))
    atomic_json(args.output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
