"""Scene groups execute different courses while feeding one pooled MAPPO rollout."""
import gc
from concurrent.futures import ThreadPoolExecutor
import torch
import warp as wp
from .course_env import CourseWarpEnv
from .course_config import COURSES,LOW_LEVEL,GAIT_COURSES,DURATIONS
from .warp_env import WarpSoccerEnv
from .gait import METRIC_NAMES
from .telemetry import active_team_reward,SKILL_NAMES


def ended_rows(env,terminated,truncated,info):
    """Compact terminal evidence before any reset mutates episode buffers."""
    done=terminated|truncated;ids=done.nonzero()
    if not ids.numel():return torch.empty((0,10),device=env.device),ids
    if env.course in LOW_LEVEL:
        w,a=ids[:,0],ids[:,1]
        rows=torch.stack((info['stable'][w,a],info['success'][w,a],
            torch.zeros_like(info['stable'][w,a]),info['fallen'][w,a],info['duration'][w,a],
            torch.zeros_like(info['duration'][w,a]),info['touch'][w,a],
            torch.zeros_like(info['stable'][w,a]),a//4),-1).float()
    elif 'team_success' in info:
        # Each contested rally supplies independent red AND blue evidence.
        w=ids[:,0].repeat_interleave(2)
        t=torch.arange(2,device=env.device).repeat(len(ids))
        rows=torch.stack((info['team_stable'][w,t],info['team_success'][w,t],info['goal'][w],
            info['team_fallen'][w,t],info['duration'][w],info['team_progress'][w,t],
            info['team_touch'][w,t],torch.zeros_like(info['duration'][w]),t),-1).float()
    else:
        w=ids[:,0]
        rows=torch.stack((info['stable'][w],info['success'][w],info['goal'][w],
            (info['fallen'][w]&env.active[w]).any(-1),info['duration'][w],info['progress'][w],
            info['touch'][w].any(-1),info['recover'][w],env.practice[w]//4),-1).float()
    return torch.cat((rows,rows.new_full((len(rows),1),COURSES.index(env.course))),-1),ids


class MixedCourseEnv:
    """Independent episode state per group; GPU streams use explicit dependencies."""
    dt=.02
    course='parallel'
    def __init__(self,count,device,seed,allocations,nconmax=1536,njmax=6144,episode_steps=None):
        self.count=count;self.device=torch.device(device)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.seed=seed;self.capacities={'nconmax':nconmax,'njmax':njmax}
        self.episode_override=episode_steps
        self.groups={};self.allocations={};self._seeded=False;self._pending=[]
        self.configure(allocations)

    def configure(self,allocations):
        if dict(allocations)==self.allocations:return
        assert sum(allocations.values())==self.count and all(n>0 for n in allocations.values())
        torch.cuda.synchronize(self.device)
        rng={name:env.generator.get_state() for name,(_,env) in self.groups.items()}
        if hasattr(self,'executor'):self.executor.shutdown(wait=True)
        self._pending=[];self.groups.clear();gc.collect();torch.cuda.empty_cache()
        self.allocations=dict(allocations)
        common=torch.cuda.Stream(device=self.device);common.wait_stream(torch.cuda.current_stream(self.device))
        offset=0
        for name in COURSES:
            if name not in allocations:continue
            count=allocations[name]
            env=CourseWarpEnv(count=count,device=str(self.device),seed=self.seed+COURSES.index(name)*1009,
                              stage=name,episode_steps=self.episode_override or DURATIONS[name],**self.capacities)
            common.wait_stream(env.torch_stream)
            if name in rng:env.generator.set_state(rng[name])
            elif self._seeded:
                seed=int(torch.randint(0,2**31-1,(1,),generator=self.generator,device=self.device).item())
                env.generator.manual_seed(seed)
            self.groups[name]=(slice(offset,offset+count),env);offset+=count
        torch.cuda.set_stream(common);self.torch_stream=common
        self.executor=ThreadPoolExecutor(max_workers=min(4,len(self.groups)))
        first=next(iter(self.groups.values()))[1]
        for name in ('observation_space','share_observation_space','action_space','roles'):
            setattr(self,name,getattr(first,name))
        self.course_ids=torch.cat([torch.full((env.count,),COURSES.index(name),device=self.device,dtype=torch.long)
                                   for name,(_,env) in self.groups.items()])
        self.active=torch.cat([env.active for _,env in self.groups.values()])
        eligible=sum(int(env.active.sum().item()) for _,env in self.groups.values())
        self.entropy_coef=sum((.005 if name=='step_in_place' else .001)*int(env.active.sum().item())
                              for name,(_,env) in self.groups.items())/eligible
        self.stat_course_ids=torch.tensor([COURSES.index(s) for s in self.groups],device=self.device)
        counts=torch.tensor([env.count for _,env in self.groups.values()],device=self.device,dtype=torch.float32)
        self.skill_weights=counts/counts.sum()
        weights=counts[:,None].expand(-1,len(METRIC_NAMES)).clone()
        gait_mask=torch.tensor([s in GAIT_COURSES for s in self.groups],device=self.device)
        for index,key in enumerate(METRIC_NAMES):
            if key in ('phase_match','velocity_error','phase_lift'):weights[:,index]*=gait_mask
        self.gait_weights=weights/weights.sum(0).clamp(min=1)
        self.course_reward_sum=torch.zeros((len(COURSES),2),device=self.device)
        self.course_skill_sum=torch.zeros((len(COURSES),len(SKILL_NAMES)),device=self.device)
        self.course_stat_sum=torch.zeros((len(COURSES),len(METRIC_NAMES)),device=self.device)
        self.reset()

    def restore_course_rng(self,states):
        for name,state in states.items():
            if name in self.groups:self.groups[name][1].generator.set_state(state.cpu())
        self._seeded=bool(states)

    def course_rng(self):return {name:env.generator.get_state() for name,(_,env) in self.groups.items()}

    def reset(self):
        if not self._seeded:
            for _,env in self.groups.values():
                seed=int(torch.randint(0,2**31-1,(1,),generator=self.generator,device=self.device).item())
                env.generator.manual_seed(seed)
            self._seeded=True
        observations=[]
        for _,env in self.groups.values():
            env.torch_stream.wait_stream(self.torch_stream)
            torch.cuda.set_stream(env.torch_stream)
            observations.append(env.reset()[0])
            self.torch_stream.wait_stream(env.torch_stream)
        torch.cuda.set_stream(self.torch_stream)
        self._obs=torch.cat(observations)
        self._pending=[]
        self.active=torch.cat([env.active for _,env in self.groups.values()])
        return self._obs,self.state(self._obs)

    def observe(self):return self._obs
    state=WarpSoccerEnv.state

    def begin_rollout(self):
        self.course_stat_sum.zero_();self.course_reward_sum.zero_();self.course_skill_sum.zero_()

    @property
    def gait_metrics(self):return self._gait_metrics
    @property
    def last_stats(self):return self._last_stats

    @torch.no_grad()
    def step(self,actions):
        observations=[];rewards=[];terminated=[];truncated=[];self._pending=[]
        for _,env in self.groups.values():env.torch_stream.wait_stream(self.torch_stream)
        def advance(item):
            name,(sl,env)=item
            torch.cuda.set_device(self.device);torch.cuda.set_stream(env.torch_stream)
            return sl,env,env.step(actions[sl])
        results=list(self.executor.map(advance,self.groups.items()))
        torch.cuda.set_stream(self.torch_stream)
        for sl,env,result in results:
            self.torch_stream.wait_stream(env.torch_stream)
            obs,_,reward,term,trunc,info=result
            observations.append(obs);rewards.append(reward)
            terminated.append(term if term.ndim==2 else term[:,None].expand(-1,8))
            truncated.append(trunc if trunc.ndim==2 else trunc[:,None].expand(-1,8))
            self._pending.append((sl,env,term,trunc,info))
        self._obs=torch.cat(observations)
        gait_matrix=torch.stack([torch.stack([env.gait_metrics[k] for k in METRIC_NAMES]) for _,env in self.groups.values()])
        self.course_stat_sum.index_add_(0,self.stat_course_ids,gait_matrix)
        self._gait_metrics=dict(zip(METRIC_NAMES,(gait_matrix*self.gait_weights).sum(0).unbind()))
        keys=SKILL_NAMES
        skill_matrix=torch.stack([torch.stack([env.last_stats[k] for k in keys]) for _,env in self.groups.values()])
        self.course_skill_sum.index_add_(0,self.stat_course_ids,skill_matrix)
        reward_matrix=torch.stack([active_team_reward(reward,env.active) for reward,(_,env) in zip(rewards,self.groups.values())])
        self.course_reward_sum.index_add_(0,self.stat_course_ids,reward_matrix)
        self._last_stats=dict(zip(keys,(skill_matrix*self.skill_weights[:,None]).sum(0).unbind()))
        return self._obs,self.state(self._obs),torch.cat(rewards),torch.cat(terminated),torch.cat(truncated),{}

    def finish_step(self):
        next_obs=self._obs.clone();records=[]
        for sl,env,term,trunc,info in self._pending:
            rows,ids=ended_rows(env,term,trunc,info)
            if len(rows):
                records.append(rows)
                env.torch_stream.wait_stream(self.torch_stream)
                torch.cuda.set_stream(env.torch_stream)
                reset_obs=(env.reset_agents(ids) if env.course in LOW_LEVEL else env.reset(ids[:,0]))[0]
                torch.cuda.set_stream(self.torch_stream)
                self.torch_stream.wait_stream(env.torch_stream)
                next_obs[sl]=reset_obs
        self._obs=next_obs;self._pending=[]
        self.active=torch.cat([env.active for _,env in self.groups.values()])
        return next_obs,torch.cat(records) if records else torch.empty((0,10),device=self.device)

    def check(self):
        for _,env in self.groups.values():env.check()

    def diagnostics(self):
        return {name:{'flags':wp.to_torch(env.failure).cpu().tolist(),
                      'overflow':wp.to_torch(env.data.overflow).cpu().tolist()}
                for name,(_,env) in self.groups.items()}
