"""Independent Q1 soccer task, strict camera-only ball input and real contacts.

No adjacent project imports or unvalidated motion CSVs. Privileged state is
kept outside actor inputs. Distance shaping uses mesh-derived foot boxes;
reset gaps are checked against MuJoCo collision geometry, contact events against
actual physics contacts. This is an experimental downstream integration.
"""
import copy
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from .dynamics import multiply
from .kinematics import Kinematics, rotate
from .observations import MotorObservation
from .physics import WarpPhysics

CONFIG = Path(__file__).resolve().parents[1]/'configs/soccer.json'
PHASES = ('touch_and_recover', 'advance_ball', 'approach_and_kick', 'score_goal')


def load_config():
    return json.loads(CONFIG.read_text())


def inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def yaw_of(q):
    return torch.atan2(2*(q[..., 0]*q[..., 3]+q[..., 1]*q[..., 2]),
                       1-2*(q[..., 2].square()+q[..., 3].square()))


class SoccerGeometry(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.nq = model, model.nq
        self.fk = Kinematics(model)
        self.ball_q = int(model.joint('ball_free').qposadr[0])
        self.ball_v = int(model.joint('ball_free').dofadr[0])
        self.ball_geom = model.geom('soccer_ball').id
        self.radius = float(model.geom_size[self.ball_geom, 0])
        self.feet = [model.body(name).id for name in ('left_ankle_roll_link','right_ankle_roll_link')]
        self.torso = model.body('torso_link').id
        self.foot_geoms, lows, highs = [], [], []
        for body in self.feet:
            identifiers = [i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_contype[i]]
            points = []
            for geom in identifiers:
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError('Expected official foot collision meshes')
                mesh = int(model.geom_dataid[geom])
                vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                matrix = np.zeros(9)
                mujoco.mju_quat2Mat(matrix,model.geom_quat[geom])
                points.append(vertices @ matrix.reshape(3,3).T + model.geom_pos[geom])
            self.foot_geoms.extend(identifiers)
            vertices = np.concatenate(points)
            lows.append(vertices.min(0)); highs.append(vertices.max(0))
        self.register_buffer('foot_low',torch.tensor(np.array(lows),dtype=torch.float32))
        self.register_buffer('foot_high',torch.tensor(np.array(highs),dtype=torch.float32))

    def gap(self,state):
        positions, rotations = self.fk(state[..., :self.nq])
        local = rotate(inverse(rotations[..., self.feet, :]),
                       state[..., self.ball_q:self.ball_q+3].unsqueeze(-2)-positions[..., self.feet,:])
        closest = local.clamp(min=self.foot_low, max=self.foot_high)
        return ((local-closest).norm(dim=-1)-self.radius).clamp_min(0).amin(-1)

    def native_gap(self,data):
        return min(mujoco.mj_geomDistance(self.model,data,self.ball_geom,geom,10.,None)
                   for geom in self.foot_geoms)

    def native_contact(self,data):
        return any(contact.dist <= 0 and self.ball_geom in contact.geom and
                   any(geom in self.foot_geoms for geom in contact.geom) and contact.efc_address >= 0
                   for contact in data.contact)


class SpawnSampler:
    """Fixed cohorts; exact sphere/convex-foot gap checked during near reset."""
    def __init__(self,model,config,seed):
        self.model,self.config = model,config
        self.geometry = SoccerGeometry(model)
        self.observe = MotorObservation(model)
        self.random = np.random.default_rng(seed)
        self.data = mujoco.MjData(model)

    def sample(self,groups):
        result, gaps = [], []
        c,m,r = self.config,self.model,self.random
        for group in np.asarray(groups):
            data = self.data
            mujoco.mj_resetData(m,data)
            data.qpos[self.observe.control.qpos_indices.numpy()] = self.observe.control.neutral.numpy()
            ball = self.geometry.ball_q
            data.qpos[ball:ball+3] = [0,0,self.geometry.radius]
            # Near cohort faces the ball; all cohorts use zero initial velocity.
            angle = r.uniform(-np.pi,np.pi)
            direction = np.array([np.cos(angle),np.sin(angle)])
            if group == 0:
                xy = r.uniform([-6.5,-4.],[6.5,4.])
                yaw = r.uniform(-np.pi,np.pi)
            elif group == 1:
                radius = c['spawn_disk_radius']*np.sqrt(r.uniform())
                xy = radius*direction
                yaw = r.uniform(-np.pi,np.pi)
            elif group == 2:
                yaw = angle
                data.qpos[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
                target = r.uniform(*c['near_foot_gap'])
                lower,upper = .05,1.5
                for _ in range(24):
                    radius = (lower+upper)/2
                    data.qpos[:2] = -radius*direction
                    mujoco.mj_forward(m,data)
                    if self.geometry.native_gap(data)<target:
                        lower=radius
                    else:
                        upper=radius
                xy = -upper*direction
            else:
                raise ValueError('Unknown spawn group')
            data.qpos[:2] = xy
            data.qpos[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
            mujoco.mj_forward(m,data)
            gap = self.geometry.native_gap(data)
            if group==2 and not c['near_foot_gap'][0]-1e-5 <= gap <= c['near_foot_gap'][1]+1e-5:
                raise RuntimeError('Near reset collision gap invalid')
            # Reject physical robot/ball overlaps, not only root proximity.
            overlaps=any(mujoco.mj_geomDistance(m,data,self.geometry.ball_geom,geom,1.,None)<.001
                         for geom in range(m.ngeom) if m.geom_bodyid[geom] and
                         geom!=self.geometry.ball_geom and m.geom_contype[geom])
            if overlaps:
                if group==2:raise RuntimeError('Robot/ball overlap in near reset')
                # Rejection uses actual collision geometry, preserving uniform
                # area sampling over valid disk/field configurations.
                replacement,replacement_gaps=self.sample(np.array([group]))
                result.append(replacement[0]);gaps.append(replacement_gaps[0]);continue
            result.append(np.r_[data.qpos,data.qvel].copy());gaps.append(gap)
        return np.array(result,dtype=np.float32),np.array(gaps)


class SoccerCamera(nn.Module):
    """PPO mounting/FOV, 25Hz detections, noise, loss and 50Hz delay queue.

    Features per frame: ball xy + detected flag, localized goal xy, yaw cos/sin.
    Vision mode masks ball truth outside FOV. Explicit truth mode bypasses
    FOV, noise, detection loss and delay for simulated privileged experiments.
    No reference target or physical ball velocity is exposed to the actor.
    """
    dimension = 7
    queue_length = 20

    def __init__(self,model,config):
        super().__init__()
        self.config = config
        self.geometry = SoccerGeometry(model)
        if config.get('goal_input_mode','vision') not in ('vision','truth'):
            raise ValueError('Unknown goal input mode')
        for key,value in [('mount',config['camera_position']),('mount_q',config['camera_quaternion_wxyz']),
                          ('optical_q',[.5,.5,.5,.5]),('goal',config['goal_xy'])]:
            self.register_buffer(key,torch.tensor(value,dtype=torch.float32))

    def truth_measurement(self,state):
        q=torch.nn.functional.normalize(state[...,3:7],dim=-1)
        ball=state[...,self.geometry.ball_q:self.geometry.ball_q+3]
        relative=rotate(inverse(q),ball-state[...,:3])[...,:2]
        return torch.cat((relative,torch.ones_like(relative[...,:1])),-1)

    def initial_features(self,state):
        measured=self.truth_measurement(state) if self.config.get('goal_input_mode','vision')=='truth' else state.new_zeros(len(state),3)
        return self.features(state,measured)

    def measurement(self,state,generator=None):
        if self.config.get('goal_input_mode','vision')=='truth':
            return self.truth_measurement(state)
        positions,rotations = self.geometry.fk(state[..., :self.geometry.nq])
        torso = self.geometry.torso
        camera_position = positions[..., torso,:]+rotate(rotations[..., torso,:],self.mount)
        camera_q = multiply(rotations[..., torso,:],self.mount_q.expand_as(rotations[...,torso,:]))
        optical_q = multiply(camera_q,self.optical_q.expand_as(camera_q))
        ball = state[..., self.geometry.ball_q:self.geometry.ball_q+3]
        optical = rotate(inverse(optical_q),ball-camera_position)
        fov = self.config['camera_fov_degrees']
        visible = ((optical[...,2]>0) & (torch.atan2(optical[...,0],optical[...,2]).abs()<np.deg2rad(fov[0]/2))
                   & (torch.atan2(optical[...,1],optical[...,2]).abs()<np.deg2rad(fov[1]/2)))
        q = torch.nn.functional.normalize(state[...,3:7],dim=-1)
        relative = rotate(inverse(q),ball-state[..., :3])[..., :2]
        distance = relative.norm(dim=-1)
        probability = (2.475-.225*distance.clamp_min(7)).clamp(0,1)
        detected = visible & (torch.rand(visible.shape,device=state.device,generator=generator)<probability)
        noise = torch.randn(relative.shape,device=state.device,generator=generator)*(
                self.config['camera_noise_base']+self.config['camera_noise_distance']*distance).unsqueeze(-1)
        measured = torch.where(detected.unsqueeze(-1),relative+noise,torch.zeros_like(relative))
        return torch.cat((measured,detected.unsqueeze(-1).to(state)),-1)

    def features(self,state,delayed_ball):
        yaw = yaw_of(torch.nn.functional.normalize(state[...,3:7],dim=-1))
        delta = self.goal-state[..., :2]
        cosine,sine = yaw.cos(),yaw.sin()
        goal = torch.stack((cosine*delta[...,0]+sine*delta[...,1],
                            -sine*delta[...,0]+cosine*delta[...,1]),-1)
        return torch.cat((delayed_ball,goal,cosine.unsqueeze(-1),sine.unsqueeze(-1)),-1)

    def advance(self,state,queue,delay,age,generator=None):
        measurement = self.measurement(state,generator)
        if self.config.get('goal_input_mode','vision')=='truth':
            queue=torch.cat((measurement.unsqueeze(-2),queue[...,:-1,:]),-2)
            return self.features(state,measurement),queue
        update = (age % 2 == 0).unsqueeze(-1)
        current = torch.where(update,measurement,queue[...,0,:])
        queue = torch.cat((current.unsqueeze(-2),queue[...,:-1,:]),-2)
        delayed = queue[torch.arange(len(state),device=state.device),delay.long()]
        return self.features(state,delayed),queue


class SoccerHead(nn.Module):
    def __init__(self,model,config):
        super().__init__()
        self.geometry=SoccerGeometry(model)
        self.config=config
        self.register_buffer('goal',torch.tensor(config['goal_xy'],dtype=torch.float32))

    def forward(self,before,after,action,previous_action,phase=0):
        g=self.geometry
        old_ball=before[...,g.ball_q:g.ball_q+2];ball=after[...,g.ball_q:g.ball_q+2]
        direction=self.goal-old_ball
        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        progress=((ball-old_ball)*direction).sum(-1)
        approach=(before[...,:2]-old_ball).norm(dim=-1)-(after[...,:2]-ball).norm(dim=-1)
        gap_gain=g.gap(before)-g.gap(after)
        q=torch.nn.functional.normalize(after[...,3:7],dim=-1)
        tilt=2*(q[...,1].square()+q[...,2].square())
        height_error=(after[...,2]-.417).abs()
        fell=(height_error>.25)|(tilt>.8)
        soft=1-torch.sigmoid((.25-height_error)/.02)*torch.sigmoid((.8-tilt)/.05)
        reward=(.005*(1-soft)+.5*gap_gain+(1.+phase)*progress+.2*approach
                -.0002*action.square().sum(-1)-.001*(action-previous_action).square().sum(-1))
        return reward,fell,soft


class EpisodeTracker:
    """Exact completed-episode counts. A timeout never becomes a fall."""
    def __init__(self,count,config,device):
        self.config=config;self.device=device
        self.records=deque(maxlen=10000)
        self.completed=0
        self.contact=torch.zeros(count,dtype=torch.bool,device=device)
        self.first_contact=torch.full((count,),-1,device=device,dtype=torch.long)
        self.goal=torch.zeros(count,dtype=torch.bool,device=device)
        self.initial_distance=torch.zeros(count,device=device)
        self.initial_gap=torch.zeros(count,device=device)
        self.start_goal_distance=torch.zeros(count,device=device)
        self.progress=torch.zeros(count,device=device)
        self.recovered=torch.zeros(count,dtype=torch.bool,device=device)

    def reset(self,ids,state,geometry,initial_gaps=None):
        self.contact[ids]=False;self.first_contact[ids]=-1;self.goal[ids]=False;self.recovered[ids]=False
        ball=state[:,geometry.ball_q:geometry.ball_q+2]
        self.initial_distance[ids]=(state[:,:2]-ball).norm(dim=-1)
        self.start_goal_distance[ids]=(ball-ball.new_tensor(self.config['goal_xy'])).norm(dim=-1)
        self.progress[ids]=0
        self.initial_gap[ids]=geometry.gap(state) if initial_gaps is None else initial_gaps

    def update(self,state,geometry,contacts,age,fell,goal,done,groups):
        first=contacts & ~self.contact
        self.first_contact[first]=age[first]
        self.contact|=contacts
        self.goal|=goal
        ball=state[:,geometry.ball_q:geometry.ball_q+2]
        self.progress=self.start_goal_distance-(ball-ball.new_tensor(self.config['goal_xy'])).norm(dim=-1)
        self.recovered|=self.contact & ~fell & (age-self.first_contact>=self.config['recovery_steps'])
        for identifier in done.nonzero().flatten().cpu().tolist():
            progress=float(self.progress[identifier])
            effective=bool(self.contact[identifier] and self.recovered[identifier] and not fell[identifier]
                           and progress>=self.config['effective_kick_progress'])
            self.records.append(dict(world=identifier,group=int(groups[identifier]),fall=bool(fell[identifier]),
                 contact=bool(self.contact[identifier]),effective_kick=effective,progress=progress,
                 approached=bool(self.initial_distance[identifier]>=1.),goal=bool(self.goal[identifier]),
                 steps=int(age[identifier]),initial_foot_gap_m=float(self.initial_gap[identifier]),
                 approach_progress_m=float(self.initial_distance[identifier]-(state[identifier,:2]-ball[identifier]).norm())))
            self.completed+=1
        return first

    def summary(self):
        records=list(self.records)
        result=dict(completed_episodes=self.completed,window_episodes=len(records))
        if records:
            result.update(effective_kick_success_rate=float(np.mean([r['effective_kick'] for r in records])),
                ball_goal_net_progress_m=float(np.mean([r['progress'] for r in records])),
                full_episode_fall_rate=float(np.mean([r['fall'] for r in records])),
                contact_rate=float(np.mean([r['contact'] for r in records])),
                goal_rate=float(np.mean([r['goal'] for r in records])))
        return result


class Curriculum:
    def __init__(self,config,phase=0):
        self.config,self.phase=config,phase

    def consider(self,records):
        """Only call with completed independent evaluation episodes."""
        c=self.config
        if self.phase>=3 or len(records)<c['promotion_min_episodes']:
            return False
        if len(records)%4 or [sum(r['group']==g for r in records) for g in range(3)]!=[len(records)//4,len(records)//4,len(records)//2]:
            return False
        group=2 if self.phase<2 else None
        selected=[r for r in records if group is None or r['group']==group]
        if not selected:
            return False
        required=c['promotion_progress_m'][self.phase]
        successes=[r['effective_kick'] and r['progress']>=required and
                   (self.phase!=2 or r['approached']) for r in selected]
        if np.mean(successes)>=c['promotion_success_rate'] and np.mean([r['fall'] for r in selected])<=c['promotion_max_fall_rate']:
            self.phase+=1
            return True
        return False


class SoccerEnvironment:
    def __init__(self,count,config,seed,device='cuda:0'):
        if count%4 or count<4:
            raise ValueError('Need a positive multiple of four for fixed 25/25/50 cohorts')
        self.config,self.device,self.count=config,torch.device(device),count
        self.physics=WarpPhysics(count,device)
        self.model=self.physics.native_model
        self.observe=MotorObservation(self.model).to(device)
        self.camera=SoccerCamera(self.model,config).to(device)
        self.head=SoccerHead(self.model,config).to(device)
        self.sampler=SpawnSampler(self.model,config,seed)
        self.generator=torch.Generator(device=device).manual_seed(seed+1)
        self.groups=torch.cat((torch.zeros(count//4),torch.ones(count//4),torch.full((count//2,),2))).long().to(device)
        self.age=torch.zeros(count,dtype=torch.long,device=device)
        self.delay=torch.zeros_like(self.age)
        self.previous=torch.zeros(count,22,device=device)
        self.history=torch.zeros(count,10,72,device=device)
        self.visual=torch.zeros(count,config['visual_history'],7,device=device)
        self.queue=torch.zeros(count,20,3,device=device)
        self.tracker=EpisodeTracker(count,config,device)
        self.phase=0
        self.reset(torch.arange(count,device=device))

    def state(self):
        return torch.cat((self.physics.qpos,self.physics.qvel),-1).clone()

    @torch.no_grad()
    def reset(self,ids):
        if not len(ids):return
        states,gaps=self.sampler.sample(self.groups[ids].cpu().numpy())
        state=torch.tensor(states,device=self.device)
        self.physics.reset_indices(ids,state[:,:self.model.nq],state[:,self.model.nq:])
        self.age[ids]=0;self.previous[ids]=0;self.queue[ids]=0
        self.delay[ids]=torch.tensor(self.sampler.random.normal(self.config['camera_delay_mean_steps'],
                        self.config['camera_delay_std_steps'],len(ids)),device=self.device).round().long().clamp(0,19)
        self.history[ids]=motor_sensor(self.observe,state,self.previous[ids],self.generator).unsqueeze(1)
        self.visual[ids]=self.camera.initial_features(state).unsqueeze(1)
        self.tracker.reset(ids,state,self.head.geometry,torch.tensor(gaps,device=self.device,dtype=state.dtype))
        return gaps

    @torch.no_grad()
    def step(self,action):
        state=self.state()
        old_history,old_visual,old_queue=self.history.clone(),self.visual.clone(),self.queue.clone()
        old_age=self.age.clone()
        contacts=torch.zeros(self.count,dtype=torch.bool,device=self.device)
        for _ in range(self.observe.control.substeps):
            self.physics.step(self.observe.control(self.physics.qpos,self.physics.qvel,action))
            contacts|=self.physics.ball_foot_contacts(self.head.geometry.ball_geom,self.head.geometry.foot_geoms)
        following=self.state()
        self.age+=1
        reward,fell,_=self.head(state,following,action,self.previous,self.phase)
        ball=following[:,self.head.geometry.ball_q:self.head.geometry.ball_q+3]
        old_ball=state[:,self.head.geometry.ball_q:self.head.geometry.ball_q+3]
        goal=(old_ball[:,0]<7)&(ball[:,0]>=7)&(ball[:,1].abs()<1.3)&(ball[:,2]<1.0)
        out=(following[:,:2].abs()>following.new_tensor([8.,5.])).any(-1)|(ball[:,:2].abs()>ball.new_tensor([8.,5.])).any(-1)
        timeout=self.age>=self.config['episode_steps']
        terminated=fell|goal|out
        done=terminated|timeout
        first=self.tracker.update(following,self.head.geometry,contacts,self.age,fell,goal,done,self.groups)
        reward=reward+.5*first.float()+10*goal.float()-fell.float()
        features,self.queue=self.camera.advance(following,self.queue,self.delay,self.age,self.generator)
        self.visual=torch.cat((self.visual[:,1:],features.unsqueeze(1)),1)
        self.history=torch.cat((self.history[:,1:],motor_sensor(self.observe,following,action,self.generator).unsqueeze(1)),1)
        self.previous=action.clone()
        transition=dict(state=state,next_state=following,action=action.clone(),boundary=done,
             terminated=terminated,context=state.new_zeros(self.count,0),history=old_history,
             visual=old_visual,queue=old_queue,delay=self.delay.clone(),age=old_age,
             reward=reward.clone(),contact=contacts,group=self.groups.clone())
        self.reset(done.nonzero().flatten())
        self.physics.check()
        return transition


def motor_sensor(observe,state,action,generator=None):
    """Encoder/IMU noise; previous executed action remains known exactly."""
    sensor=observe(state,action)
    scales=sensor.new_tensor([.01]*22+[.1]*22+[.02]*3+[.01]*3+[0.]*22)
    return sensor+torch.randn(sensor.shape,device=state.device,generator=generator)*scales
