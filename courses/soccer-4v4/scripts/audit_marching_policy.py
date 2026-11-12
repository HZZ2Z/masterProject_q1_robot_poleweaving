"""Compare a saved actor under matched marching commands and target observations.

Read-only replay: no optimizer updates and no production checkpoint writes.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import torch
from q1_4v4.course_env import CourseWarpEnv
from q1_4v4.course_config import DURATIONS
from q1_4v4.learning import make_args,make_team,load_team,with_team_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--case',choices=('legacy_observation_2cm','legacy_observation_high','origin_observation_2cm','origin_observation_high','legacy_observation_final','origin_observation_final','zero_action'),required=True)
    options=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(options.checkpoint,map_location='cuda:0',weights_only=False)
    legacy=options.case.startswith('legacy')
    env=CourseWarpEnv(count=8,stage='step_in_place',seed=42,episode_steps=DURATIONS['step_in_place'],marching_target_observation='legacy' if legacy else 'origin')
    env.generator.manual_seed(42);env.reset()
    if options.case.endswith('2cm'):env.gait_height.fill_(.02)
    elif options.case.endswith('high'):env.gait_height.copy_(.03+2*(env.gait_height-.02))
    args=make_args();args.hidden_size=ck['args']['hidden_size']
    args.marching_exploration_revision=ck.get('marching_exploration_revision',0)
    args.marching_exploration_mix=ck.get('marching_exploration_mix',.25)
    policy,trainer,_=make_team(args,env,env.device,gpu_rollout=True,shared=True)
    load_team(ck['teams'][0],policy,trainer);trainer.prep_rollout()
    alive=torch.ones((8,8),device=env.device,dtype=torch.bool);falls=torch.zeros_like(alive)
    saved={k:torch.zeros((8,8),device=env.device) for k in ('end_drift','max_drift','max_heading','left_steps','right_steps','alternations','knee','cross_ratio')}
    total={k:0. for k in ('width','drift','reward')};count=0
    identities=(torch.arange(8,device=env.device)//4).expand(8,8).reshape(-1)
    rnn=torch.zeros((64,1,args.hidden_size),device=env.device);ones=torch.ones((64,1),device=env.device)
    with torch.no_grad():
        for step in range(600):
            obs=with_team_identity(env.observe().reshape(64,-1),identities)
            action=torch.zeros((8,8,22),device=env.device) if options.case=='zero_action' else policy.act(obs,rnn,ones,deterministic=True)[0].reshape(8,8,22)
            _,_,reward,term,trunc,_=env.step(action)
            pos=env.xpos[:,env.bodies];rot=env.xmat[:,env.bodies].reshape(8,8,3,3)
            feetlocal=torch.einsum('nafi,naij->nafj',env.xpos[:,env.feet]-pos[:,:,None],rot)
            drift=torch.linalg.vector_norm(pos[:,:,:2]-env.start_positions[:,:,:2],dim=-1)
            for key,value in {'width':feetlocal[:,:,0,1]-feetlocal[:,:,1,1],'drift':drift,'reward':reward.squeeze(-1)}.items():total[key]+=value[alive].sum().item()
            count+=alive.sum().item();falls|=term&alive
            evidence={'end_drift':drift,'max_drift':env.gait_drift_max,'max_heading':env.gait_heading_max,
                'left_steps':env.gait_landings[:,:,0],'right_steps':env.gait_landings[:,:,1],
                'alternations':env.gait_switches,'knee':env.gait_knee_credit/env.agent_steps.clamp(min=1),
                'cross_ratio':env.gait_cross/env.agent_steps.clamp(min=1)}
            for key,value in evidence.items():saved[key]=torch.where(alive,value,saved[key])
            alive&=~(term|trunc)
            if not alive.any():break
            ended=(term|trunc).nonzero()
            if len(ended):
                env.reset_agents(ended)
                if options.case.endswith('2cm'):env.gait_height[ended[:,0],ended[:,1]]=.02
                elif options.case.endswith('high'):
                    w,a=ended[:,0],ended[:,1];env.gait_height[w,a]=.03+2*(env.gait_height[w,a]-.02)
    env.check()
    result={'case':options.case,'checkpoint_iteration':ck['iteration'],'robots':64,'steps':step+1,'episode_limit_seconds':env.episode_steps*env.dt,
            'fall_rate':falls.float().mean().item(),'observation_mode':env.marching_target_observation,
            'source_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[3]/name).read_bytes()).hexdigest() for name in ('courses/soccer-4v4/q1_4v4/gait.py','courses/soccer-4v4/q1_4v4/course_env.py','shared/q1-mujoco/resources/Q1.yaml')},'means':{k:v/count for k,v in total.items()},
            'episode':{k:{'mean':v.mean().item(),'max':v.max().item()} for k,v in saved.items()}}
    Path(options.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
