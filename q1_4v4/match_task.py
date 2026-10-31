"""Continuous 4v4 rallies, individual respawns, and an auditable reward ledger."""
import torch
import weakref
from collections import deque


class MatchControl:
    """Certify progressively longer kickoff approaches using disjoint rally evidence."""
    distances=((.23,.31),(.45,.65),(.8,1.1),(1.3,1.6),(1.8,2.4))
    counters=('near_red','near_blue','touch_red','touch_blue','advance_red','advance_blue',
              'quality_red','quality_blue','standard_red','standard_blue',
              'standard_touch_red','standard_touch_blue','standard_goal_red','standard_goal_blue')

    def __init__(self,state=None):
        state=state or {}
        self.amplitude=float(state.get('amplitude',.45))
        self.windows=int(state.get('windows',0))
        self.level=int(state.get('level',0));self.updates=int(state.get('updates',0))
        self.evidence=dict.fromkeys(self.counters,0.);self.evidence.update(state.get('evidence',{}))
        self.history=deque(state.get('history',[]),maxlen=32)
        self.reason=state.get('reason','samples')
        if not .45<=self.amplitude<=1. or self.windows<0 or not 0<=self.level<=4:
            raise ValueError('Invalid match control state')

    def update(self,standing,respawns,evidence=None):
        evidence={key:float((evidence or {}).get(key,0.)) for key in self.counters}
        self.history.append(evidence);self.updates+=1
        for key,value in evidence.items():self.evidence[key]+=value
        if max(respawns)>12.:
            self.amplitude=max(.45,round(self.amplitude-.025,6));self.windows=0
        if self.updates%16:return
        if min(self.evidence['near_'+team] for team in ('red','blue'))<256:
            self.reason='samples';return
        rates=self._rates(self.evidence)
        self.reason=('posture' if min(standing)<.90 or max(respawns)>3. or
                     min(rates['quality_'+team] for team in ('red','blue'))<.90 else
                     'touch' if min(rates['touch_'+team] for team in ('red','blue'))<.65 else
                     'advance' if min(rates['advance_'+team] for team in ('red','blue'))<.40 else
                     'goals' if self.level==4 and min((rates['standard_goal_'+team] or 0.) for team in ('red','blue'))<.05
                     else 'pass')
        self.windows=self.windows+1 if self.reason=='pass' else 0
        self.evidence=dict.fromkeys(self.counters,0.)
        if self.windows>=3:
            self.level=min(4,self.level+1)
            self.amplitude=min(1.,round(self.amplitude+.025,6))
            self.windows=0;self.reason='expanded'
            self.history.clear()

    @staticmethod
    def _rates(counts):
        rates={}
        for team in ('red','blue'):
            for key in ('touch','advance','quality'):
                n=counts['near_'+team];rates[key+'_'+team]=counts[key+'_'+team]/n if n else None
            for key in ('standard_touch','standard_goal'):
                n=counts['standard_'+team];rates[key+'_'+team]=counts[key+'_'+team]/n if n else None
        return rates

    def rates(self):
        counts={key:sum(row.get(key,0.) for row in self.history) for key in self.counters}
        return self._rates(counts)

    def state_dict(self):
        return dict(amplitude=self.amplitude,windows=self.windows,level=self.level,updates=self.updates,
                    evidence=self.evidence,history=list(self.history),reason=self.reason)


class MatchTask:
    discount=.99
    terms=('survival','upright','height','joint_velocity','action_smooth','foot_slip',
           'nonfoot_ground','approach','role_progress','role_position','ball_potential','foot_contact',
           'goal_for','goal_against','robot_reset','ball_out','nonfoot_ball',
           'stance_width','approach_distance','attack_milestone')
    metrics=('standing_red','standing_blue','respawns_red_per_min','respawns_blue_per_min',
             'falls_red_per_min','falls_blue_per_min','reward_residual',
             'chaser_distance_red','chaser_distance_blue')+MatchControl.counters

    def __init__(self,env):
        self.env=weakref.proxy(env)
        n,d=env.count,env.device
        self.failures=torch.zeros((n,2),device=d,dtype=torch.long)
        self.falls=torch.zeros((n,2),device=d,dtype=torch.long)
        self.foot_valid=torch.zeros(n,device=d,dtype=torch.bool)
        self.has_foot=torch.zeros(n,device=d,dtype=torch.bool)
        self.advance_origin=torch.zeros((n,8),device=d)
        self.owner=torch.full((n,),-1,device=d,dtype=torch.long)
        self.spawn=torch.zeros((n,8,2),device=d)
        self.standard=torch.zeros(n,device=d,dtype=torch.bool)
        self.kickoff_team=torch.zeros(n,device=d,dtype=torch.long)
        self.spawn_level=torch.zeros(n,device=d,dtype=torch.long)
        self.quality_steps=torch.zeros((n,2),device=d)
        self.attack_paid=torch.zeros((n,2),device=d,dtype=torch.bool)
        self.last_metrics=torch.zeros(len(self.metrics),device=d)
        self.last_terms=torch.zeros((len(self.terms),2),device=d)
        self.last_residual=torch.zeros((),device=d)
        self.control=MatchControl()

    def reset_worlds(self,ids):
        e=self.env
        self.failures[ids]=0;self.falls[ids]=0;self.foot_valid[ids]=False;self.owner[ids]=-1
        self.has_foot[ids]=False;self.advance_origin[ids]=0
        self.quality_steps[ids]=0;self.attack_paid[ids]=False
        self.spawn[ids]=e.qpos[ids][:,e.rootq[:,None]+torch.arange(2,device=e.device)]
        self.update_targets()

    def spawn_curriculum(self,ids):
        """Keep the ball at midfield and all eight players; shorten the robot approach."""
        e=self.env;n=len(ids);level=self.control.level
        layout=getattr(e,'match_layout','mixed')
        standard=(ids%10==0) if layout=='mixed' else torch.full((n,),layout=='standard',device=e.device)
        if level==4:standard=torch.ones_like(standard)
        self.standard[ids]=standard;self.spawn_level[ids]=level
        flip=torch.randint(2,(),device=e.device,generator=e.generator)
        self.kickoff_team[ids]=(ids+ids//10+flip)%2
        selected=ids[~standard]
        if not len(selected):return
        n=len(selected);team=self.kickoff_team[selected]
        role=2+torch.randint(2,(n,),device=e.device,generator=e.generator)
        attacker=team*4+role;opponent=(1-team)*4+role
        lo,hi=self.control.distances[level]
        for robot,extra in ((attacker,0.),(opponent,max(.6,.9-.1*level))):
            sign=e.sign[robot]
            distance=e.uniform((n,),lo+extra,hi+extra)
            foot_sign=torch.where(e.uniform((n,))>.5,1.,-1.)
            e.qpos[selected,e.rootq[robot]]=e.ball[selected,0]-sign*distance
            e.qpos[selected,e.rootq[robot]+1]=e.ball[selected,1]+sign*foot_sign*.09+e.uniform((n,),-.015,.015)

    @staticmethod
    def posture(up,height,width,bad_contact):
        """Bounded, continuous posture shaping keeps a crouch from earning survival."""
        width_error=(width-.1754).abs().sub(.05).clamp(min=0)
        height_x=((height-.417)/.045).square()
        width_x=(width_error/.075).square()
        quality=((up-.8)/.15).clamp(0,1)*torch.exp(-((height-.417)/.06).square()
                    -(width_error/.075).square())*(~bad_contact).float()
        healthy=(up>.85)&(height>.36)&(height<.50)&(width>.09)&(width<.30)&~bad_contact
        return quality,healthy,height_x/(1+height_x),width_x/(1+width_x)

    def reset_agents(self,w,a):
        # Each new individual trajectory starts with zero potential. Otherwise a
        # negative inherited potential could pay repeatedly on deliberate falls.
        self.advance_origin[w,a]=self.env.ball[w,0].clamp(-7.1,7.1)

    def respawn_positions(self,w,a):
        """Prefer the original spawn; alternative lanes avoid occupied points and the ball."""
        e=self.env;n=len(w)
        primary=self.spawn[w,a]
        offsets=e.tensor([[0,0],[.15,0],[-.15,0],[0,.15],[0,-.15]])
        candidates=primary[:,None]+offsets[None]
        sign=e.sign[a]
        lanes=sign*(-3.6+2.4*(a%4).float())
        back=e.tensor([-6.5,-5.5,-4.5,-3.0,-1.5])[None]*sign[:,None]
        fallback=torch.stack((back,lanes[:,None].expand(-1,5)),-1)
        candidates=torch.cat((candidates,fallback),1)
        positions=e.qpos[w][:,e.rootq[:,None]+torch.arange(2,device=e.device)]
        dist=torch.linalg.vector_norm(candidates[:,:,None]-positions[:,None],dim=-1)
        dist=dist.masked_fill(torch.arange(8,device=e.device)[None,None]==a[:,None,None],float('inf'))
        clearance=torch.minimum(dist.amin(-1)-.65,
            torch.linalg.vector_norm(candidates-e.ball[w,None,:2],dim=-1)-.45)
        safe=clearance>=0
        first=safe.long().argmax(-1)
        index=torch.where(safe.any(-1),first,clearance.argmax(-1))
        return candidates[torch.arange(n,device=e.device),index]

    @staticmethod
    def role_targets(ball,positions,active,sign):
        """One active field player chases; poor posture cannot discard responsibility."""
        norm=lambda v:torch.linalg.vector_norm(v,dim=-1)
        distances=norm(positions[:,:,:2]-ball[:,None,:2]).reshape(-1,2,4)
        eligible=active.reshape(-1,2,4).clone();eligible[:,:,0]=False
        selected=distances.masked_fill(~eligible,float('inf')).argmin(-1)
        chase=torch.nn.functional.one_hot(selected,4).bool()&eligible.any(-1)[:,:,None]
        chase=chase.reshape(-1,8)
        anchors=ball[:,None,:2].expand(-1,8,-1).clone()
        anchors[:,[0,4],0]=-6*sign[[0,4]]
        anchors[:,[0,4],1]=ball[:,1:2].clamp(-1,1)
        anchors[:,[1,5],0]=-3*sign[[1,5]]
        anchors[:,[1,5],1]=ball[:,1:2]*.5
        anchors[:,[2,6],0]-=.8*sign[[2,6]]
        anchors[:,[2,6],1]-=.8*sign[[2,6]]
        anchors[:,[3,7],0]-=1.2*sign[[3,7]]
        anchors[:,[3,7],1]+=1.5*sign[[3,7]]
        anchors=torch.where(chase[:,:,None],ball[:,None,:2],anchors)
        anchors[:,:,0].clamp_(-6.7,6.7);anchors[:,:,1].clamp_(-4.,4.)
        return anchors,chase

    def update_targets(self):
        e=self.env;pos=e.xpos[:,e.bodies]
        anchors,_=self.role_targets(e.ball,pos,e.active,e.sign)
        e.targets[:,:,:2]=anchors;e.targets[:,:,2]=.417

    def observations(self,features):
        e=self.env;teams=torch.arange(8,device=e.device)[None]//4
        own=self.owner[:,None]==teams
        other=(self.owner[:,None]>=0)&~own
        features=features.clone()
        features[:,:,:3]=torch.stack((own,other,self.foot_valid[:,None].expand(-1,8)),-1).float()
        features[:,:,3]=(self.kickoff_team[:,None]==teams).float()
        features[:,:,4]=self.spawn_level[:,None].float()/4.
        features[:,:,5]=self.advance_origin*e.sign[None]*self.has_foot[:,None]/7.1
        features[:,:,6]=1.+self.has_foot[:,None].float()
        return features

    @staticmethod
    def potential_reward(before,after,terminal,discount=.99):
        return discount*torch.where(terminal,0.,after)-before

    def step(self,action,last_action,before,pos_before,base):
        e=self.env;norm=lambda v:torch.linalg.vector_norm(v,dim=-1)
        pos=e.xpos[:,e.bodies];rot=e.xmat[:,e.bodies].reshape(e.count,8,3,3)
        up=rot[:,:,2,2]
        fallen=(pos[:,:,2]<.2)|(up<.3)
        outside=(pos[:,:,0].abs()>7.7)|(pos[:,:,1].abs()>4.7)
        robot_reset=(fallen|outside)&e.active
        feetlocal=torch.einsum('nafi,naij->nafj',e.xpos[:,e.feet]-pos[:,:,None],rot)
        width=feetlocal[:,:,0,1]-feetlocal[:,:,1,1]
        quality,healthy,height_cost,width_cost=self.posture(up,pos[:,:,2],width,e.gait_bad_contact.bool())
        ball_healthy=(up>.8)&(pos[:,:,2]>.3)&(pos[:,:,2]<.55)&~e.gait_bad_contact.bool()
        movement_quality=(.25+.75*quality)*ball_healthy
        self.quality_steps+=healthy.reshape(e.count,2,4).float().mean(-1)
        phi_before=e.sign[None]*(before[:,0:1].clamp(-7.1,7.1)-self.advance_origin)*self.has_foot[:,None]
        foot=e.foot_hits.bool().any(-1)&ball_healthy&~e.nonfoot_ball_hits.bool()
        anyfoot=foot.any(-1)
        body=e.nonfoot_ball_hits.bool().any(-1)
        first_trajectory=anyfoot&~body&~self.has_foot
        self.advance_origin=torch.where(first_trajectory[:,None],before[:,0:1].clamp(-7.1,7.1),self.advance_origin)
        self.has_foot|=anyfoot&~body
        self.foot_valid=torch.where(body,False,torch.where(anyfoot,True,self.foot_valid))
        touched=e.touches.bool().reshape(e.count,2,4).any(-1)
        unique=touched.sum(-1)==1
        owner=touched.long().argmax(-1)
        self.owner=torch.where(touched.any(-1),torch.where(unique,owner,-1),self.owner)
        first_foot=foot&~e.ever_feet.any(-1)
        e.ever_feet|=e.foot_hits.bool()&foot[:,:,None]
        self.failures+=robot_reset.reshape(e.count,2,4).sum(-1)
        self.falls+=(fallen&e.active).reshape(e.count,2,4).sum(-1)
        world_term=(base['goal']!=0)|base['out']
        world_trunc=(e.steps>=e.episode_steps)&~world_term
        terminated=robot_reset|world_term[:,None]
        truncated=world_trunc[:,None].expand(-1,8)&~terminated
        anchors,chase=self.role_targets(before,pos_before,e.active,e.sign)
        delta_ball_distance=norm(pos_before[:,:,:2]-before[:,None,:2])-norm(pos[:,:,:2]-before[:,None,:2])
        ball_distance=norm(pos[:,:,:2]-e.ball[:,None,:2])
        role_delta=norm(pos_before[:,:,:2]-anchors)-norm(pos[:,:,:2]-anchors)
        phi_after=e.sign[None]*(e.ball[:,0:1].clamp(-7.1,7.1)-self.advance_origin)*self.has_foot[:,None]
        winner=base['goal'][:,None]*e.sign[None]>0
        conceded=base['goal'][:,None]*e.sign[None]<0
        legit_goal=(base['goal']!=0)&self.foot_valid
        team_touch=e.ever_feet.any(-1).reshape(e.count,2,4).any(-1)
        team_delta=(e.ball[:,0]-e.start_ball[:,0])[:,None]*e.tensor([1.,-1.])
        e.team_progress=torch.maximum(e.team_progress,torch.where(team_touch&self.foot_valid[:,None],team_delta,0.))
        milestone=(e.team_progress>=.10)&team_touch&~self.attack_paid
        self.attack_paid|=milestone
        dt=e.dt
        components={
            'survival':dt*.2*quality,
            'upright':-dt*2*(1-up.clamp(-1,1)),
            'height':-dt*.6*height_cost,
            'joint_velocity':-dt*.02*e.qvel[:,e.vidx].square().mean(-1).clamp(max=25),
            'action_smooth':-dt*.1*(torch.tanh(action)-last_action).square().mean(-1),
            'foot_slip':-dt*.1*((e.foot_slip/.05).square().clamp(max=4)*e.gait_contact.bool()).mean(-1),
            'nonfoot_ground':-dt*.3*e.gait_bad_contact.bool().float(),
            'approach':4.*(delta_ball_distance.clamp(0,.05)*movement_quality+delta_ball_distance.clamp(-.05,0))*chase,
            'role_progress':.3*role_delta.clamp(-.05,.05)*(~chase)*movement_quality,
            'role_position':-dt*.03*norm(pos[:,:,:2]-anchors).clamp(max=8)*(~chase),
            'ball_potential':self.potential_reward(phi_before,phi_after,terminated,self.discount),
            'foot_contact':2.*first_foot,
            'goal_for':20.*winner*legit_goal[:,None],
            'goal_against':-20.*conceded,
            'robot_reset':-5.*robot_reset,
            'ball_out':-1.*base['out'][:,None]*(self.owner[:,None]==torch.arange(8,device=e.device)[None]//4),
            'nonfoot_ball':-.05*e.nonfoot_ball_hits.bool().float(),
            'stance_width':-dt*.4*width_cost,
            'approach_distance':-dt*.4*(ball_distance-.25).clamp(0,3)*chase,
            'attack_milestone':milestone.repeat_interleave(4,-1).float(),
        }
        reward=torch.stack([components[k] for k in self.terms]).sum(0)*e.active
        self.last_terms=torch.stack([(components[k]*e.active).reshape(e.count,2,4).mean((0,2)) for k in self.terms])
        self.last_residual=(reward-torch.stack(list(components.values())).sum(0)*e.active).abs().max()
        standing=healthy.reshape(e.count,2,4).float().mean((0,2))
        respawns=robot_reset.reshape(e.count,2,4).float().mean((0,2))*60/dt
        falls=(fallen&e.active).reshape(e.count,2,4).float().mean((0,2))*60/dt
        team_success=(base['goal'][:,None]*e.tensor([1.,-1.])>0)&legit_goal[:,None]
        if e.course=='teamplay':
            contacts=e.ever_feet.any(-1).reshape(e.count,2,4).sum(-1)
            team_success|=(contacts>=2)&(e.team_progress>=1.)
            team_success&=self.standard[:,None]
        team_quality=self.quality_steps/e.steps[:,None].clamp(min=1)
        team_stable=(self.failures==0)&(team_quality>=.90)&((e.steps[:,None]*dt>=5.)|legit_goal[:,None])
        ended=world_term|world_trunc
        current_level=self.spawn_level==self.control.level
        near=(~self.standard|(self.control.level==4))&current_level&ended
        owner=torch.nn.functional.one_hot(self.kickoff_team,2).bool()
        eligible=near[:,None]&owner
        standard=self.standard[:,None]&ended[:,None]
        counts={}
        for index,team in enumerate(('red','blue')):
            own=eligible[:,index]
            counts['near_'+team]=own.float().sum()
            counts['touch_'+team]=(own&team_touch[:,index]).float().sum()
            counts['advance_'+team]=(own&self.attack_paid[:,index]).float().sum()
            counts['quality_'+team]=(own&team_stable[:,index]).float().sum()
            counts['standard_'+team]=standard[:,0].float().sum()
            counts['standard_touch_'+team]=(standard[:,0]&team_touch[:,index]).float().sum()
            counts['standard_goal_'+team]=(standard[:,0]&team_success[:,index]).float().sum()
        chaser_distance=(ball_distance*chase).reshape(e.count,2,4).sum(-1).mean(0)
        self.last_metrics=torch.cat((standing,respawns,falls,self.last_residual[None],chaser_distance,
                                    torch.stack([counts[key] for key in MatchControl.counters])))
        info=dict(stage=e.course,goal=base['goal'],out=base['out'],fallen=fallen,
            stable=team_stable.any(-1),success=team_success.any(-1),duration=e.steps*dt,
            progress=e.team_progress.max(-1).values,touch=e.ever_feet.any(-1),recover=torch.zeros(e.count,device=e.device,dtype=torch.bool),
            team_stable=team_stable,team_success=team_success,team_fallen=self.falls>0,
            team_progress=e.team_progress,team_touch=team_touch,
            world_terminated=world_term,world_truncated=world_trunc,robot_resets=robot_reset)
        zero=torch.zeros((),device=e.device)
        e.forward_kick.zero_();e.side_kick.zero_()
        e.last_stats=dict(upright=up.mean(),height=pos[:,:,2].mean(),forward=zero,side=zero,
            task_distance=norm(pos[:,:,:2]-e.ball[:,None,:2]).mean(),
            left=e.foot_hits[:,:,0].float().mean(),right=e.foot_hits[:,:,1].float().mean())
        self.update_targets()
        obs=e.observe()
        return obs,e.state(obs),reward[:,:,None],terminated,truncated,info
