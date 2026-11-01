"""Skill curricula layered over the shared 8-Q1 Warp physical backend."""
import torch
import warp as wp
from .warp_env import WarpSoccerEnv
from .course_config import COURSES, LOW_LEVEL, KICKS, FAMILIES, AMPLITUDE, command, GAIT_COURSES, CONTESTS, FULL_MATCHES
from . import gait
from .match_task import MatchTask


@wp.kernel
def feet_contact(n: wp.array[int], geoms: wp.array[wp.vec2i], worlds: wp.array[int],
                 dist: wp.array[float], owner: wp.array[int], robot_owner: wp.array[int], ball: int,
                 hits: wp.array3d[int], nonfoot: wp.array2d[int]):
    c = wp.tid()
    if c < n[0] and dist[c] <= 0.:
        a,b = geoms[c][0], geoms[c][1]
        g = int(-1)
        if a == ball: g = b
        elif b == ball: g = a
        if g >= 0 and owner[g] >= 0:
            k = owner[g]
            wp.atomic_max(hits,worlds[c],k//2,k%2,1)
        elif g >= 0 and robot_owner[g] >= 0:
            wp.atomic_max(nonfoot,worlds[c],robot_owner[g],1)


class CourseWarpEnv(WarpSoccerEnv):
    def __init__(self, count=64, device='cuda:0', seed=42, stage='balance', marching_target_observation='origin', match_layout='mixed', **kwargs):
        self.course = stage
        if match_layout not in ('mixed','curriculum','standard'):raise ValueError('Unknown match layout')
        self.match_layout=match_layout
        if marching_target_observation not in ('legacy','origin'):raise ValueError('Unknown marching target observation mode')
        self.marching_target_observation=marching_target_observation
        self.count=count; self.device=torch.device(device)
        self.ready = False
        # Kernels are captured by parent constructor; buffers must exist before _substep.
        self.foot_hits = torch.zeros((count,8,2),device=device,dtype=torch.int32)
        self.nonfoot_ball_hits = torch.zeros((count,8),device=device,dtype=torch.int32)
        self.foot_owner = None
        gait.initialize(self)
        super().__init__(count,device,seed,FAMILIES[stage],**kwargs)
        self.agent_steps = torch.zeros((count,8),device=self.device,dtype=torch.long)
        self.heading = torch.zeros((count,8),device=self.device)
        self.ever_feet = torch.zeros((count,8,2),device=self.device,dtype=torch.bool)
        self.foot_displacement = torch.zeros((count,8,2),device=self.device)
        self.forward_total = torch.zeros_like(self.foot_displacement)
        self.side_total = torch.zeros_like(self.foot_displacement)
        self.control_steps = torch.zeros(count,device=self.device,dtype=torch.long)
        self.hold_steps = torch.zeros_like(self.agent_steps)
        self.forward_kick = torch.zeros((count,8),device=self.device)
        self.side_kick = torch.zeros_like(self.forward_kick)
        self.contact_at = torch.full_like(self.agent_steps,-1)
        self.progress = torch.zeros(count,device=self.device)
        self.team_progress = torch.zeros((count,2),device=self.device)
        self.received = torch.zeros(count,device=self.device,dtype=torch.bool)
        self.aligned = torch.zeros(count,device=self.device,dtype=torch.bool)
        self.active = torch.ones((count,8),device=self.device,dtype=torch.bool)
        gait.allocate(self)
        from gymnasium.spaces import Box
        import numpy as np
        self.observation_space=Box(-np.inf,np.inf,(135,),dtype=np.float32)
        self.share_observation_space=Box(-np.inf,np.inf,(1084,),dtype=np.float32)
        self.ready = True
        self.configure(stage)
        self.reset()

    def _substep(self):
        if self.foot_owner is None:
            # Build mapping before first capture. Only the two foot collision bodies count.
            m = self.model
            owner = torch.full((m.ngeom,),-1,device=self.device,dtype=torch.int32)
            from .scene import PREFIXES
            self.feet = torch.tensor([[m.body(p+s+'_ankle_roll_link').id for s in ('left','right')] for p in PREFIXES],device=self.device)
            for a in range(8):
                for f in range(2):
                    owner[torch.as_tensor(m.geom_bodyid == int(self.feet[a,f]),device=self.device)] = a*2+f
            self.foot_owner_tensor = owner
            self.foot_owner = wp.from_torch(owner)
        wp.launch(gait.clear_ground,dim=(self.count,8,2),inputs=[wp.from_torch(self.gait_contact),
            wp.from_torch(self.gait_contact_index)],device=self.wp_device)
        super()._substep()
        wp.launch(feet_contact,dim=self.data.contact.geom.shape[0],inputs=[self.data.nacon,
            self.data.contact.geom,self.data.contact.worldid,self.data.contact.dist,self.foot_owner,self.owner_wp,
            self.reference.ballgeom,wp.from_torch(self.foot_hits),wp.from_torch(self.nonfoot_ball_hits)],device=self.wp_device)
        wp.launch(gait.ground_contacts,dim=self.data.contact.geom.shape[0],inputs=[self.data.nacon,
            self.data.contact.geom,self.data.contact.worldid,self.data.contact.dist,self.foot_owner,self.owner_wp,
            self.model.geom('floor').id,wp.from_torch(self.gait_contact),wp.from_torch(self.gait_contact_index),
            wp.from_torch(self.gait_bad_contact)],device=self.wp_device)
        wp.launch(gait.ground_points,dim=(self.count,8,2),inputs=[wp.from_torch(self.gait_contact_index),
            self.data.contact.pos,wp.from_torch(self.gait_point)],device=self.wp_device)

    def configure(self, stage):
        self.course = stage; self.stage = FAMILIES[stage]
        self.course_amplitude = AMPLITUDE[stage]
        self.active.fill_(True)
        if stage in FULL_MATCHES:
            if not hasattr(self,'match_task'):self.match_task=MatchTask(self)
            self.course_amplitude=self.match_task.control.amplitude

    def _participants(self, ids):
        """One ball drill per world; identities are resampled only at rally reset."""
        n=len(ids)
        if self.course in LOW_LEVEL:return
        if self.course not in CONTESTS:
            # Randomized balanced identity assignment across a reset batch.
            p=(torch.arange(n,device=self.device)+torch.randint(8,(),device=self.device,generator=self.generator))%8
            p=(p%2)*4+p//2  # interleave teams even for small world batches
            p=p[torch.randperm(n,device=self.device,generator=self.generator)]
            if self.course=='goalkeeper':p=(p//4)*4
            self.practice[ids]=p
        p=self.practice[ids];team=p//4
        mask=torch.zeros((n,8),device=self.device,dtype=torch.bool)
        rows=torch.arange(n,device=self.device)
        if self.course in ('teamplay','match'):mask.fill_(True)
        elif self.course in ('duel','attack_2v1','game_2v2'):
            mask[:,[2,6]]=True
            if self.course=='attack_2v1':mask[rows,team*4+3]=True
            if self.course=='game_2v2':mask[:,[3,7]]=True
        else:
            mask[rows,p]=True
            if self.course=='pass_receive':mask[rows,team*4+(p%4+1)%4]=True
        self.active[ids]=mask
        # Park unused bodies outside the playable region; their falls cannot end a drill.
        w,a=(~mask).nonzero(as_tuple=True);world=ids[w]
        self.qpos[world,self.rootq[a]]=0.
        self.qpos[world,self.rootq[a]+1]=7.+a.float()*.7
        self.qvel[world[:,None],self.rootv[a,None]+torch.arange(6,device=self.device)]=0
        self.qvel[world[:,None],self.vidx[a]]=0

    def reset(self, ids=None):
        result = super().reset(ids)
        if not self.ready: return result
        ids = self.rows if ids is None else ids
        self._participants(ids)
        self.team_progress[ids]=0
        self.agent_steps[ids] = 0; self.ever_feet[ids] = False; self.contact_at[ids] = -1
        self.forward_total[ids] = 0; self.side_total[ids] = 0; self.control_steps[ids] = 0; self.hold_steps[ids] = 0
        self.progress[ids] = 0; self.received[ids] = False; self.aligned[ids] = False
        self.heading[ids] = torch.where(self.sign>0,0.,torch.pi)
        self._targets(ids)
        p=self.practice[ids];sign=self.sign[p];n=len(ids)
        if self.course in CONTESTS:
            # Symmetric midfield kickoff; the sampled reporting identity has no spawn advantage.
            formation=self.tensor([[-5.8,0],[-3.8,0],[-1.8,-1.5],[-1.8,1.5]]).repeat(2,1)*self.sign[:,None]
            positions=formation[None]+self.uniform((n,8,2),-.08,.08)
            w,a=self.active[ids].nonzero(as_tuple=True)
            self.qpos[ids[w],self.rootq[a]]=positions[w,a,0]
            self.qpos[ids[w],self.rootq[a]+1]=positions[w,a,1]
            self.qpos[ids,self.ballq:self.ballq+2]=self.uniform((n,2),-.2,.2)
        elif self.course=='goalkeeper':
            self.qpos[ids,self.ballq]=-sign*self.uniform((n,),2.8,4.5)
            self.qpos[ids,self.ballq+1]=self.uniform((n,),-.8,.8)
            speed=self.uniform((n,),1.,2.5)
            self.qvel[ids,self.ballv]=-sign*speed
            target_y=self.uniform((n,),-.9,.9)
            flight=(6.-self.qpos[ids,self.ballq].abs())/speed
            self.qvel[ids,self.ballv+1]=(target_y-self.qpos[ids,self.ballq+1])/flight
            self.qpos[ids,self.rootq[p]]=-sign*self.uniform((n,),5.6,6.)
            self.qpos[ids,self.rootq[p]+1]=self.uniform((n,),-.25,.25)
        elif self.course not in LOW_LEVEL:
            ball_xy=torch.stack((self.uniform((n,),-3.5,3.5),self.uniform((n,),-2.3,2.3)),-1)
            self.qpos[ids,self.ballq:self.ballq+2]=ball_xy
            distance=self.uniform((n,),.8,1.3) if self.course=='approach' else self.uniform((n,),.65,.95) if self.course=='align' else self.uniform((n,),.29,.36)
            foot_offset=-.1 if self.course.endswith('left') else .1
            if self.course.startswith('side_'):foot_offset=-foot_offset
            lateral=sign*(self.uniform((n,),-.12,.12) if self.course in ('approach','align') else self.uniform((n,),-.02,.02)+foot_offset)
            self.qpos[ids,self.rootq[p]]=ball_xy[:,0]-sign*distance
            self.qpos[ids,self.rootq[p]+1]=ball_xy[:,1]+lateral
            yaw=torch.where(p<4,0.,torch.pi)
            if self.course.startswith('side_'):
                yaw+=-torch.pi/2 if self.course=='side_left' else torch.pi/2
            spread=.6 if self.course=='approach' else .9 if self.course=='align' else .08
            yaw+=self.uniform((n,),-spread,spread)
            self.qpos[ids,self.rootq[p]+3]=torch.cos(yaw/2)
            self.qpos[ids,self.rootq[p]+6]=torch.sin(yaw/2)
            if self.course=='pass_receive':
                receiver=(p//4)*4+(p%4+1)%4
                self.qpos[ids,self.rootq[receiver]]=ball_xy[:,0]+sign*self.uniform((n,),1.5,2.5)
                self.qpos[ids,self.rootq[receiver]+1]=ball_xy[:,1]+self.uniform((n,),-.7,.7)
                receiver_yaw=torch.where(p<4,torch.pi,0.)+self.uniform((n,),-.2,.2)
                self.qpos[ids,self.rootq[receiver]+3]=torch.cos(receiver_yaw/2)
                self.qpos[ids,self.rootq[receiver]+6]=torch.sin(receiver_yaw/2)
        if self.course in FULL_MATCHES:self.match_task.spawn_curriculum(ids)
        self.start_ball[ids] = self.ball[ids]
        self.start_positions[ids] = self.qpos[ids][:,self.rootq[:,None]+torch.arange(3,device=self.device)]
        self.foot_hits[ids] = 0
        self.nonfoot_ball_hits[ids] = 0
        w=ids[:,None].expand(-1,8).flatten();a=torch.arange(8,device=self.device).expand(len(ids),-1).flatten()
        gait.reset(self,w,a)
        self.forward()
        if self.course in FULL_MATCHES:self.match_task.reset_worlds(ids)
        return self.observe(),self.state()

    def _targets(self, ids):
        n = len(ids)
        if self.course=='step_in_place' and self.marching_target_observation=='origin':self.targets[ids]=self.start_positions[ids]
        if self.course == 'backward': self.targets[ids,:,0] = self.start_positions[ids,:,0]-self.sign*.8
        if self.course == 'lateral':
            self.targets[ids,:,:2] = self.start_positions[ids,:,:2]
            self.targets[ids,:,1] += torch.where(self.uniform((n,8))>.5,.8,-.8)
        if self.course == 'turn':
            self.heading[ids] += torch.where(self.uniform((n,8))>.5,1.,-1.)*self.uniform((n,8),.5,1.5)
        if self.course == 'stop': self.targets[ids,:,:2] = self.start_positions[ids,:,:2]+self.sign[None,:,None]*self.tensor([.6,0])

    def reset_agents(self, pairs):
        if not len(pairs): return self.observe(),self.state()
        w,a = pairs[:,0],pairs[:,1]
        spawn=self.match_task.respawn_positions(w,a) if self.course in FULL_MATCHES else None
        # Only reset the indicated articulated robot; ball and other robots retain their trajectories.
        q0 = self.tensor(self.model.qpos0)
        qindices = self.rootq[a,None]+torch.arange(7,device=self.device)
        self.qpos[w[:,None],qindices] = q0[qindices]
        if spawn is not None:
            self.qpos[w,self.rootq[a]]=spawn[:,0]
            self.qpos[w,self.rootq[a]+1]=spawn[:,1]
        self.qpos[w[:,None],self.qidx[a]] = self.neutral
        self.qvel[w[:,None],self.rootv[a,None]+torch.arange(6,device=self.device)] = 0
        self.qvel[w[:,None],self.vidx[a]] = 0
        wp.to_torch(self.data.qacc_warmstart)[w[:,None],self.rootv[a,None]+torch.arange(6,device=self.device)] = 0
        wp.to_torch(self.data.qacc_warmstart)[w[:,None],self.vidx[a]] = 0
        self.hold_steps[w,a] = 0; self.previous_action[w,a] = 0;self.agent_steps[w,a] = 0;self.reached[w,a] = False
        self.start_positions[w,a] = self.qpos[w[:,None],self.rootq[a,None]+torch.arange(3,device=self.device)]
        self.targets[w,a] = self.start_positions[w,a]
        dx = -.8 if self.course=='backward' else .8
        if self.course!='step_in_place' or self.marching_target_observation=='legacy':self.targets[w,a,0] += self.sign[a]*dx
        if self.course=='lateral':
            self.targets[w,a,0] = self.start_positions[w,a,0]
            self.targets[w,a,1] += torch.where(self.uniform((len(w),))>.5,.8,-.8)
        self.heading[w,a] = torch.where(a<4,0.,torch.pi)
        if self.course=='turn': self.heading[w,a] += self.uniform((len(w),),-.9,.9)
        gait.reset(self,w,a)
        self.forward()
        if self.course in FULL_MATCHES:
            self.match_task.reset_agents(w,a)
            self.match_task.update_targets()
        return self.observe(),self.state()

    def observe(self):
        obs = super().observe()
        if not self.ready: return obs
        obs[:,:,109:115] = self.tensor(command(self.course))
        obs[:,:,-2] = self.agent_steps/self.episode_steps if self.course in LOW_LEVEL else self.steps[:,None]/self.episode_steps
        obs[:,:,-1] = self.course_amplitude
        if self.course=='turn':
            rot=self.xmat[:,self.bodies].reshape(self.count,8,3,3)
            yaw=torch.atan2(rot[:,:,1,0],rot[:,:,0,0]);error=self.heading-yaw
            obs[:,:,106:109] = torch.stack((torch.cos(error),torch.sin(error),torch.zeros_like(error)),-1)
        gait_obs=gait.observations(self)
        match_obs=self.match_task.observations(torch.zeros_like(gait_obs)) if self.course in FULL_MATCHES else torch.zeros_like(gait_obs)
        return torch.cat((obs[:,:,:115],gait_obs,match_obs,obs[:,:,115:]),-1)

    @torch.no_grad()
    def step(self, action):
        from .curriculum import STAGES,AMPLITUDES
        controlled = action*self.active[:,:,None]
        self.foot_hits.zero_()
        self.nonfoot_ball_hits.zero_()
        self.gait_bad_contact.zero_()
        last_action = self.previous_action.clone()
        foot_before = self.xpos[:,self.feet].clone()
        before = self.ball.clone()
        pos_before = self.xpos[:,self.bodies].clone()
        if self.course=='balance_push':
            force=wp.to_torch(self.data.xfrc_applied)
            pulse=(self.agent_steps % 100 == 50).float()
            force[:,self.bodies,:2] = self.uniform((self.count,8,2),-8,8)*pulse[:,:,None]
        _,_,_,term,trunc,base = super().step(controlled)
        self.agent_steps += 1
        gait_reward,gait_objective,gait_quality=gait.measure(self,foot_before)
        if self.course in FULL_MATCHES:return self.match_task.step(action,last_action,before,pos_before,base)
        pos=self.xpos[:,self.bodies]; rot=self.xmat[:,self.bodies].reshape(self.count,8,3,3)
        norm=lambda x:torch.linalg.vector_norm(x,dim=-1)
        upright=rot[:,:,2,2]
        fallen=(pos[:,:,2]<.2)|(upright<.3)
        outside=(pos[:,:,0].abs()>7.7)|(pos[:,:,1].abs()>4.7)
        reward=self.dt*(1-2*(1-upright)-10*(pos[:,:,2]-.417).square()-.02*self.qvel[:,self.vidx].square().mean(-1))
        reward -= self.dt*.1*(torch.tanh(action)-last_action).square().mean(-1)
        self.ever_feet |= self.foot_hits.bool()
        first=self.foot_hits.any(-1)&(self.contact_at<0)
        self.contact_at[first]=self.agent_steps[first]
        direction=torch.stack((self.sign*7,torch.zeros_like(self.sign)),-1)[None]-before[:,None,:2]
        direction=direction/norm(direction).clamp(min=1e-6)[:,:,None]
        ball_delta=self.ball-before
        progress=(ball_delta[:,None,:2]*direction).sum(-1).clamp(min=0,max=.04)
        speed=(self.xpos[:,self.feet]-foot_before)/self.dt
        local=torch.einsum('nafi,naij->nafj',speed,rot)
        mag=local[:,:,:,0].abs()+local[:,:,:,1].abs()
        valid=self.foot_hits.bool()&(mag>.05)&(upright[:,:,None]>.8)&~fallen[:,:,None]
        forward=progress[:,:,None]*valid*local[:,:,:,0].clamp(min=0)/mag.clamp(min=1e-6)
        side=progress[:,:,None]*valid*local[:,:,:,1].abs()/mag.clamp(min=1e-6)
        self.forward_total += forward; self.side_total += side
        selected_foot=(forward+side).argmax(-1,keepdim=True)
        self.forward_kick=forward.gather(-1,selected_foot).squeeze(-1)
        self.side_kick=side.gather(-1,selected_foot).squeeze(-1)
        reward += 2*self.forward_kick+8*self.side_kick
        targetdist=norm(pos[:,:,:2]-self.targets[:,:,:2])
        prevdist=norm(pos_before[:,:,:2]-self.targets[:,:,:2])
        yaw=torch.atan2(rot[:,:,1,0],rot[:,:,0,0]); yawerr=torch.atan2(torch.sin(self.heading-yaw),torch.cos(self.heading-yaw)).abs()
        low_success=torch.zeros_like(fallen)
        if self.course in ('balance','balance_push'): low_success.fill_(True)
        if self.course in ('move','backward','lateral','stop'):
            reward += 3*(prevdist-targetdist).clamp(-.05,.05)-self.dt*.2*targetdist.clamp(max=2)
            self.reached |= targetdist<.25
            low_success=self.reached
            if self.course=='stop':
                velocity=norm(self.qvel[:,self.rootv[: ,None]+torch.arange(2,device=self.device)])
                low_success=(targetdist<.25)&(velocity<.1)
        if self.course=='turn':
            reward += self.dt*(torch.cos(self.heading-yaw)-.5*yawerr)
            low_success=yawerr<.2
        if self.course in ('turn','stop'):
            self.hold_steps = torch.where(low_success,self.hold_steps+1,0)
            low_success &= self.hold_steps*self.dt >= 1.
        ball_distance=norm(pos[:,:,:2]-self.ball[:,None,:2])
        p=self.practice
        if self.course not in LOW_LEVEL:
            reward += .3*(norm(pos_before[:,:,:2]-before[:,None,:2])-ball_distance).clamp(-.05,.05)
            reward += self.sign*ball_delta[:,0:1].clamp(-.2,.2)*2+self.sign*base['goal'][:,None]*20
        if self.course in ('duel','attack_2v1','game_2v2','teamplay','match'):
            anchors=self.ball[:,None,:2].expand(-1,8,-1).clone()
            anchors[:,[0,4],0]=-6*self.sign[[0,4]]
            anchors[:,[0,4],1]=self.ball[:,1:2].clamp(-1,1)
            anchors[:,[1,5],0]=-3*self.sign[[1,5]]
            anchors[:,[1,5],1]=self.ball[:,1:2]*.5
            anchors[:,[3,7],0]-=self.sign[[3,7]]
            anchors[:,[3,7],1]+=1.5
            reward-=self.dt*.03*norm(pos[:,:,:2]-anchors).clamp(max=8)*self.active
        practice_contact=self.ever_feet[self.rows,p].any(-1)
        self.progress=torch.maximum(self.progress,torch.where(practice_contact,self.sign[p]*(self.ball[:,0]-self.start_ball[:,0]),0.))
        heading_goal=torch.atan2(-pos[self.rows,p,1],self.sign[p]*7-pos[self.rows,p,0])
        alignerr=torch.atan2(torch.sin(heading_goal-yaw[self.rows,p]),torch.cos(heading_goal-yaw[self.rows,p])).abs()
        behind=(self.sign[p]*(self.ball[:,0]-pos[self.rows,p,0])>.1)
        aligned=(ball_distance[self.rows,p]<.65)&(alignerr<.25)&behind
        self.aligned |= aligned
        if self.course=='align':
            self.hold_steps[self.rows,p]=torch.where(aligned,self.hold_steps[self.rows,p]+1,0)
        if self.course=='align': reward[self.rows,p] += self.dt*(torch.cos(heading_goal-yaw[self.rows,p])-ball_distance[self.rows,p])
        selected=0 if self.course.endswith('left') else 1
        designated=self.ever_feet[self.rows,p,selected]
        recover=practice_contact&((self.agent_steps[self.rows,p]-self.contact_at[self.rows,p])*self.dt>=1.5)&~fallen[self.rows,p]
        receiver=(p//4)*4+(p%4+1)%4
        receiver_dist=norm(self.ball[:,:2]-pos[self.rows,receiver,:2])
        receiver_touch=self.ever_feet[self.rows,receiver].any(-1)
        self.received |= self.foot_hits[self.rows,receiver].any(-1)&practice_contact&(
            self.contact_at[self.rows,receiver]>self.contact_at[self.rows,p])
        if self.course=='pass_receive':
            # Pass aim is receiver, rather than incidental two-player contact anywhere.
            receiver_progress=norm(before[:,:2]-pos[self.rows,receiver,:2])-receiver_dist
            reward[self.rows,p] += 4*receiver_progress.clamp(-.05,.05)
            reward[self.rows,receiver] += self.dt*(receiver_dist<.5).float()
        ball_speed=norm(self.qvel[:,self.ballv:self.ballv+2])
        self.control_steps = torch.where((ball_distance[self.rows,p]<.8)&(ball_speed<1.)&practice_contact,self.control_steps+1,0)
        forward_ok=self.forward_total[self.rows,p,selected]>.02
        side_ok=self.side_total[self.rows,p,selected]>.02
        designated &= side_ok if self.course.startswith('side_') else forward_ok
        if self.course.startswith('side_'):
            designated &= self.side_total[self.rows,p,selected] > self.forward_total[self.rows,p,selected]
        elif self.course in ('kick_left','kick_right'):
            designated &= self.forward_total[self.rows,p,selected] > self.side_total[self.rows,p,selected]
        team_touch=(self.ever_feet.any(-1)&self.active).reshape(self.count,2,4).any(-1)
        team_delta=(self.ball[:,0]-self.start_ball[:,0])[:,None]*self.tensor([1.,-1.])
        self.team_progress=torch.maximum(self.team_progress,torch.where(team_touch,team_delta,0.))
        scoring=(base['goal'][:,None]*self.tensor([1.,-1.])>0)&team_touch
        touched_goal=(base['goal']!=0)&team_touch.any(-1)
        if self.course=='goalkeeper':
            keeper=torch.where(p<4,0,4)
            objective=self.ever_feet[self.rows,keeper].any(-1)&(base['goal']==0)&(self.ball[:,0].abs()<6.9)
            reward[self.rows,keeper] += self.dt*(6.9-self.ball[:,0].abs()).clamp(-1,2)
        else:
            objective={'approach':ball_distance[self.rows,p]<.45,'align':self.hold_steps[self.rows,p]*self.dt>=.5,
                'kick_left':designated&(self.progress>=.3),'kick_right':designated&(self.progress>=.3),
                'side_left':designated&(self.progress>=.3),'side_right':designated&(self.progress>=.3),
                'kick_recover':(self.progress>=.3)&recover&(self.gait_landings[self.rows,p].sum(-1)>=2),
                'dribble':(self.progress>=1.5)&(self.control_steps*self.dt>=2.),
                'pass_receive':self.received&(self.control_steps>=25)}.get(self.course,torch.zeros(self.count,device=self.device,dtype=torch.bool))
        if self.course=='pass_receive':
            caught=self.received&(receiver_dist<.5)&(ball_speed<.6)
            self.control_steps=torch.where(caught,self.control_steps+1,0)
            objective=caught&(self.control_steps*self.dt>=.5)
        if self.course in GAIT_COURSES or self.course=='weight_shift':
            reward+=gait_reward
        if self.course in ('weight_shift','step_in_place','slow_walk','velocity_track','walk_turn','start_stop'):
            low_success=gait_objective
        elif self.course in ('move','backward','lateral','turn','stop'):
            low_success &= gait_quality
        elif self.course=='approach':
            objective &= gait_quality[self.rows,p]
        reward -= 5*(fallen|outside)
        reward*=self.active
        if self.course in LOW_LEVEL:
            terminated=fallen|outside
            truncated=(self.agent_steps>=self.episode_steps)&~terminated
            stable=~terminated&(self.agent_steps*self.dt>=5.)
            success=stable&low_success
            duration=self.agent_steps*self.dt
        else:
            active_fall=(fallen&self.active).any(-1)
            terminated=(base['goal']!=0)|base['out']|active_fall|(outside&self.active).any(-1)
            truncated=(self.steps>=self.episode_steps)&~terminated
            stable=~active_fall&~(outside&self.active).any(-1)&((self.steps*self.dt>=5.)|touched_goal)
            success=stable&objective
            duration=self.steps*self.dt
        active_mean=lambda x:(x*self.active).sum()/self.active.sum().clamp(min=1)
        self.last_stats={'upright':active_mean(upright),'height':active_mean(pos[:,:,2]),'forward':active_mean(self.forward_kick),
                         'side':active_mean(self.side_kick),'task_distance':(receiver_dist if self.course=='pass_receive' else ball_distance[self.rows,p]).mean(),
                         'left':active_mean(self.foot_hits[:,:,0].float()),
                         'right':active_mean(self.foot_hits[:,:,1].float())}
        info=dict(stage=self.course,goal=base['goal'],fallen=fallen,stable=stable,success=success,
                  duration=duration,progress=self.progress,touch=self.ever_feet.any(-1),recover=recover,
                  out=base['out'])
        if self.course in CONTESTS:
            team_fall=(fallen&self.active).reshape(self.count,2,4).any(-1)
            team_out=(outside&self.active).reshape(self.count,2,4).any(-1)
            team_stable=~team_fall&~team_out&((self.steps[:,None]*self.dt>=5.)|touched_goal[:,None])
            objective=scoring
            if self.course=='teamplay':
                contacts=(self.ever_feet.any(-1)&self.active).reshape(self.count,2,4).sum(-1)
                objective=scoring|((self.team_progress>=1.)&(contacts>=2))
            info.update(stable=team_stable.any(-1),success=(team_stable&objective).any(-1),
                        team_stable=team_stable,team_success=team_stable&objective,
                        team_fallen=team_fall,team_progress=self.team_progress,
                        team_touch=team_touch)
        obs=self.observe()
        return obs,self.state(obs),reward[:,:,None],terminated,truncated,info
