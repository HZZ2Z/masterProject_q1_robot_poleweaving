"""Supervised dynamics, analytic model policy gradients and real GAE values."""
import torch
from torch import nn
import torch.nn.functional as F
from supermimic.models import mlp, frozen
from utils.torch_math import quat_rotate_inverse, quat_mul, quat_rotate


class TemporalReplay:
    def __init__(self, steps, envs, dim, device):
        self.capacity, self.count, self.cursor = steps, 0, 0
        self.states = torch.empty(steps, envs, dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(steps, envs, 22, device=device)
        self.rewards = torch.empty(steps, envs, device=device)
        self.done = torch.empty(steps, envs, device=device)
        self.fall = torch.empty_like(self.done)

    def add(self, state, action, target, reward, done, fall):
        for buffer, value in zip((self.states, self.actions, self.next_states,
                                  self.rewards, self.done, self.fall),
                                 (state, action, target, reward, done, fall)):
            buffer[self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def sample(self, batch, length, validation=False):
        if self.count < length:
            raise ValueError('Not enough consecutive transitions')
        device = self.states.device
        start = torch.randint(self.count - length + 1, (batch,), device=device)
        oldest = self.cursor if self.count == self.capacity else 0
        times = (oldest + start[:, None] + torch.arange(length, device=device)) % self.capacity
        count = self.states.shape[1]
        held_out = max(1, count // 8) if count >= 8 else 0
        low, high = (count - held_out, count) if validation and held_out else (0, count - held_out)
        worlds = torch.randint(low, high, (batch, 1), device=device)
        return tuple(x[times, worlds] for x in (self.states, self.actions,
                     self.next_states, self.rewards, self.done, self.fall))


class SoccerAgent(nn.Module):
    def __init__(self, dim, obs_dim=79, width=256, horizon=4, lr=2e-4, history_dim=0, model_width=None):
        super().__init__()
        self.dim, self.obs_dim, self.validation_horizon = dim, obs_dim, horizon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('action_limit', torch.tensor(1.))
        self.register_buffer('model_action_scale', torch.full((22,), .05))
        self.register_buffer('dynamic_mask', torch.ones(dim))
        self.register_buffer('model_indices', torch.empty(0, dtype=torch.long))
        self.register_buffer('model_mean', torch.empty(0))
        self.register_buffer('model_scale', torch.empty(0))
        self.register_buffer('velocity_delta_scale', torch.empty(0))
        self.model_width = model_width or width
        self.policy_width = width
        self.value_domain_positions = []
        self.history_dim = history_dim
        self.layout = None
        self.last_diagnostics = {}
        self.dynamics = mlp(dim + 22, (width, width, width), dim)
        self.outcomes = mlp(dim + 22 + dim, (width, width), 2)
        self.actor = mlp(obs_dim + history_dim, (width, width), 22)
        self.critic = mlp(dim, (width, width), 1)
        nn.init.normal_(self.dynamics[-1].weight, std=.001)
        nn.init.zeros_(self.dynamics[-1].bias)
        # Begin near the validated neutral stance, with exploration in real rollouts.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.outcomes[-1].bias[1] = -4.
        self.lr = lr
        self.model_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def make_optimizers(self):
        self.model_optimizer = torch.optim.Adam(
            list(self.dynamics.parameters()) + list(self.outcomes.parameters()), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas=(.9, .95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    @torch.no_grad()
    def calibrate(self, states, actions=None, targets=None):
        self.mean.copy_(states.mean(0))
        self.scale.copy_(states.std(0, unbiased=False).clamp_min(.1))
        if self.model_indices.numel():
            features = self.physical_features(states)
            self.model_mean.copy_(features.mean(0))
            self.model_scale.copy_(features.std(0, unbiased=False).clamp_min(.001))
            v, end = self.layout['velocity_start'], self.layout['velocity_end']
            delta_scale = ((targets[:,v:end]-states[:,v:end]).square().mean(0).sqrt()
                           if targets is not None else .1*self.scale[v:end])
            self.velocity_delta_scale.copy_(delta_scale.clamp_min(.001))
        if actions is not None:
            self.model_action_scale.copy_(actions.std(0, unbiased=False).clamp(.025, .25))
        self.ensure_ball_motion_scales()

    @torch.no_grad()
    def ensure_ball_motion_scales(self, resuming=False):
        """Static warmup must not make contact speeds thousands of model units.

        Input bounds cover ordinary ball motion (20 m/s, 200 rad/s), even when
        warmup never moves the ball. On resume, preserve output units but refit
        input sensitivities: expanding formerly clipped inputs while multiplying
        their weights can destabilize autoregressive predictions.
        """
        if not self.layout or 'ball_velocity' not in self.layout or not self.model_indices.numel():
            return False
        l = self.layout
        floors = self.model_scale.new_zeros(self.model_scale.shape)
        for offset, minimum in enumerate((1., 1., 1., 10., 10., 10.)):
            floors[self.model_indices == l['ball_velocity']+offset] = minimum
        for offset in range(3):
            floors[self.model_indices == l['ball_start']+offset] = 1.
        new_input = torch.maximum(self.model_scale, floors)
        new_delta = self.velocity_delta_scale.clone()
        start = l['ball_velocity']-l['velocity_start']
        new_delta[start:start+6] = torch.maximum(new_delta[start:start+6], new_delta.new_tensor([.1,.1,.1,1.,1.,1.]))
        changed = not (torch.equal(new_input,self.model_scale) and torch.equal(new_delta,self.velocity_delta_scale))
        if changed and resuming:
            ratio = self.velocity_delta_scale/new_delta
            self.dynamics[-1].weight.mul_(ratio[:,None])
            self.dynamics[-1].bias.mul_(ratio)
            if self.model_optimizer is not None:
                for parameter in (self.dynamics[0].weight, self.outcomes[0].weight,
                                  self.dynamics[-1].weight, self.dynamics[-1].bias):
                    self.model_optimizer.state.pop(parameter, None)
        self.model_scale.copy_(new_input)
        self.velocity_delta_scale.copy_(new_delta)
        return changed

    def normalize(self, state):
        # Physical state must remain invertible: a moving ball can leave the
        # warmup range. Only policy observations are clipped, below.
        return (state - self.mean) / self.scale

    def policy_observation(self, normalized, mean=None, scale=None):
        mean = self.mean if mean is None else mean
        scale = self.scale if scale is None else scale
        obs = normalized[..., :self.obs_dim].clamp(-20,20)
        history = normalized[..., -self.history_dim:].clamp(-20,20) if self.history_dim else None
        if self.layout and self.layout.get('policy_absolute_joints'):
            # Encoder angles plus the known commanded neutral pose, never ball
            # ground truth. Absolute proprioception distinguishes reset stances.
            defaults = normalized[...,368:390]*scale[368:390]+mean[368:390]
            factor = self.layout['normalization']['dof_pos']
            obs = obs.clone()
            obs[...,13:35] = (obs[...,13:35]*scale[13:35]+mean[13:35])/factor+defaults
            if history is not None:
                raw = history*scale[-self.history_dim:]+mean[-self.history_dim:]
                raw = raw.reshape(*raw.shape[:-1],-1,self.obs_dim)
                history = history.clone().reshape_as(raw)
                history[...,13:35] = raw[...,13:35]/factor+defaults.unsqueeze(-2)
                history = history.flatten(-2)
        return torch.cat((obs,history),-1) if history is not None else obs

    def act(self, state):
        return self.action_limit * self.actor(self.policy_observation(self.normalize(state))).tanh()

    def physical_features(self, raw):
        # Flat-ground dynamics is invariant to translating robot and ball together.
        raw = raw.clone()
        b, r = self.layout['ball_start'], self.layout['root_xy']
        raw[..., b:b+2] = raw[..., b:b+2] - raw[..., r:r+2]
        return raw[..., self.model_indices]

    def value_features(self, normalized):
        if not self.model_indices.numel():
            return normalized
        raw = normalized*self.scale+self.mean
        l = self.layout
        p, q, v, b = l['progress_index'],l['root_quat'],l['ball_velocity'],l['ball_start']
        progress = raw[:,p:p+1].clamp(0,1)
        quat = raw[:,q:q+4][:,[1,2,3,0]]
        ball_velocity = quat_rotate_inverse(quat, raw[:,v:v+3])
        # Goal and stall information cannot create fictitious standing rewards
        # associated with an arbitrary absolute field location or camera noise.
        context = torch.cat((progress,raw[:,p+1:p+7]*progress,
                             raw[:,p+11:p+16]*progress,raw[:,p-1:p]*progress),-1)
        co,si = raw[:,90:91],raw[:,91:92]
        root = raw[:,l['root_xy']:l['root_xy']+2]
        anchors = []
        for start in (p+7,p+9):
            delta = raw[:,start:start+2]-root
            anchors.append(torch.cat((co*delta[:,:1]+si*delta[:,1:],
                                      -si*delta[:,:1]+co*delta[:,1:]),-1)*progress)
        physical = ((self.physical_features(raw)-self.model_mean)/self.model_scale).clamp(-20,20)
        domain = physical[:,self.value_domain_positions]
        return torch.cat((raw[:,79:85],raw[:,92:158],raw[:,162:166],raw[:,368:390],
                          raw[:,85:87],raw[:,88:90]*progress,raw[:,b+2:b+3],
                          ball_velocity,context,*anchors,domain),-1).clamp(-20,20)

    def value(self, normalized):
        return self.critic(self.value_features(normalized))

    def predict(self, state, action, visual_noise=None):
        normalized_action = action / self.model_action_scale
        if self.model_indices.numel():
            features = ((self.physical_features(state*self.scale+self.mean)-self.model_mean) / self.model_scale).clamp(-20,20)
            delta = self.dynamics(torch.cat((features, normalized_action), -1))
            following = state.clone()
            start, end = self.layout['velocity_start'], self.layout['velocity_end']
            following[..., start:end] = state[..., start:end] + delta*self.velocity_delta_scale/self.scale[start:end]
        else:
            following = state + self.dynamic_mask * self.dynamics(torch.cat((state, normalized_action), -1))
        if self.layout is not None:
            # Previous-action observation is deterministic, not a free model output.
            raw = following * self.scale + self.mean
            raw = raw.clone()
            raw[..., 57:79] = action
            raw[..., 136:158] = action
            if self.layout.get('integrate_positions'):
                raw = self.integrate_positions(state*self.scale+self.mean, raw)
            q = self.layout['root_quat']
            quat = raw[..., q:q+4].clone()
            raw[..., q:q+4] = F.normalize(quat, dim=-1)
            if 'joint_velocity' in self.layout:
                raw = self.project_observations(state*self.scale+self.mean, raw, action, visual_noise=visual_noise)
            if self.history_dim:
                history = state[..., -self.history_dim:] * self.scale[-self.history_dim:] + self.mean[-self.history_dim:]
                raw[..., -self.history_dim:] = torch.cat((history[..., self.obs_dim:], raw[..., :self.obs_dim]), -1)
            following = (raw - self.mean) / self.scale
        if self.model_indices.numel():
            p, end = self.layout['physical_start'], self.layout['physical_end']
            ce = self.layout['context_end']
            outcome_input = torch.cat((features, normalized_action, following[...,p:end], state[...,ce-18:ce]), -1)
        else:
            outcome_input = torch.cat((state, normalized_action, following), -1)
        outcomes = self.outcomes(outcome_input.detach() if self.model_indices.numel() else outcome_input)
        return following, outcomes[..., 0], outcomes[..., 1]

    def integrate_positions(self, current, predicted):
        """Trapezoidal position integration with local-angular quaternion updates.

        Predicting tiny pose deltas independently of velocity produced inaccurate
        action derivatives even when aggregate physical validation looked good.
        """
        l, n = self.layout, predicted.clone()
        dt = l['dt']
        for position, velocity in ((l['root_xy'], l['root_velocity']),
                                   (l['ball_start'], l['ball_velocity'])):
            v = .5*(current[:,velocity:velocity+6] + predicted[:,velocity:velocity+6])
            n[:,position:position+3] = current[:,position:position+3] + dt*v[:,:3]
            angle = dt*v[:,3:].norm(dim=-1, keepdim=True)
            vector = .5*dt*torch.sinc(angle/(2*torch.pi))*v[:,3:]
            delta = torch.cat((vector, (angle/2).cos()), -1)
            q = current[:,position+3:position+7][:,[1,2,3,0]]
            following = F.normalize(quat_mul(q, delta), dim=-1)
            n[:,position+3:position+7] = following[:,[3,0,1,2]]
        j, v = l['joints_start'], l['joint_velocity']
        n[:,j:j+22] = current[:,j:j+22] + .5*dt*(current[:,v:v+22]+predicted[:,v:v+22])
        return n

    def project_observations(self, current, predicted, action, visual_noise=None):
        """Derive proprioception/privileged observations from physical state.

        Camera geometry determines visibility; policy rollouts reuse sampled
        detection and measurement noise across line-search candidates. Delayed
        measurements pass through the same queue as real observations.
        """
        l, n = self.layout, predicted.clone()
        def features(raw):
            raw = raw.clone()
            q = raw[:, l['root_quat']:l['root_quat']+4]
            quat = q[:, [1,2,3,0]]
            gravity = quat_rotate_inverse(quat, raw.new_tensor([0.,0.,-1.]).expand(len(raw), -1))
            yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))
            co, si = yaw.cos(), yaw.sin()
            xy = raw[:, l['root_xy']:l['root_xy']+2]
            def relative(target):
                delta = target-xy
                return torch.stack((co*delta[:,0]+si*delta[:,1], -si*delta[:,0]+co*delta[:,1]), -1)
            goal = relative(raw.new_tensor([7.,0.]))
            ball = relative(raw[:, l['ball_start']:l['ball_start']+2])
            v, j, d = l['root_velocity'], l['joints_start'], l['default_start']
            angular = raw[:,v+3:v+6]*l['normalization']['ang_vel']
            joints = (raw[:,j:j+22]-raw[:,d:d+22])*l['normalization']['dof_pos']
            velocity = raw[:,l['joint_velocity']:l['joint_velocity']+22]*l['normalization']['dof_vel']
            clean = torch.cat((gravity*l['normalization']['gravity'], angular, goal,
                               co[:,None],si[:,None],joints,velocity), -1)
            linear = quat_rotate_inverse(quat, raw[:,v:v+3])*l['normalization']['lin_vel']
            return clean, ball, linear
        old, old_ball, _ = features(current)
        clean, ball, linear = features(n)
        # Actor proprioception indices exclude the delayed ball triplet.
        residual = (torch.cat((current[:,:6],current[:,9:57]), -1)-old).detach()
        sensed = clean + residual
        p = l['progress_index']
        elapsed = current[:,p-2] + l['dt']/60
        visible = current[:,87:88].detach().clamp(0,1)
        queue = current[:,308:368].reshape(-1,20,3)
        residual = (queue[:,0,:2]-old_ball*queue[:,0,2:3]).detach()
        if 'camera' in l:
            geometric = self.camera_visibility(n)[:,None]
            if visual_noise is not None:
                detection_probability = torch.where(geometric,
                    (2.475-.225*ball.norm(dim=-1,keepdim=True).clamp_min(7)).clamp(0,1),
                    .01*(ball[:,:1]>-2).float())
                visible = (visual_noise[:,2:3] < detection_probability).float()
                residual = visual_noise[:,:2]*(.149+.124*ball.norm(dim=-1,keepdim=True))
            else:
                old_geometric = self.camera_visibility(current)[:,None]
                visible = torch.where(geometric,torch.where(old_geometric,visible,1.),
                                      torch.where(old_geometric,0.,visible))
        sample = torch.cat(((ball+residual)*visible, visible), -1)
        update = (torch.round(elapsed*60/l['dt']).long() % 2 == 0)[:,None]
        sample = torch.where(update, sample, queue[:,0])
        queue = torch.cat((sample[:,None],queue[:,:-1]), 1)
        delay = (current[:,307]*10).round().long().clamp(0,19)
        vision = queue[torch.arange(len(n),device=n.device),delay]
        n[:,:79] = torch.cat((sensed[:,:6],vision,sensed[:,6:],action), -1)
        # 91 privileged coordinates use exact physical features, not a free head.
        n[:,79:170] = torch.cat((clean[:,:6],ball,visible,clean[:,6:],action,
                                current[:,158:162],linear,n[:,l['root_z']:l['root_z']+1],
                                current[:,166:168],n[:,l['ball_velocity']:l['ball_velocity']+2]), -1)
        n[:,308:368] = queue.flatten(1)
        targets = current[:,368:390] + action*action.new_tensor(l['action_scale'])
        n[:,240:262] = torch.where((current[:,306]*10 < l['decimation'])[:,None],targets,current[:,240:262])
        n[:,p-2] = elapsed
        goal = (n[:,l['ball_start']] > 7) & (n[:,l['ball_start']+1].abs() < 1.3)
        n[:,p-1] = torch.where(goal, current[:,p-1]+1/50, 0.)
        if 'football_feedback' in l:
            n = self.project_task_context(current, n)
        return n

    def camera_ball(self, raw):
        """Ball coordinates in the same optical frame as the real PPO camera."""
        l, cfg = self.layout, self.layout['camera']
        q, r, b, j = l['root_quat'], l['root_xy'], l['ball_start'], l['joints_start']
        root_quat = raw[:,q:q+4][:,[1,2,3,0]]
        position, rotation = self.camera_kinematics(raw[:,j:j+22])[cfg['frame']]
        mount_quat = quat_mul(root_quat,rotation)
        mount_position = raw[:,r:r+3]+quat_rotate(root_quat,position)
        camera_position = mount_position+quat_rotate(mount_quat,raw.new_tensor(cfg['position']).expand(len(raw),-1))
        camera_quat = quat_mul(mount_quat,raw.new_tensor(cfg['rotation']).expand(len(raw),-1))
        optical_quat = quat_mul(camera_quat,raw.new_tensor([.5,.5,.5,.5]).expand(len(raw),-1))
        return quat_rotate_inverse(optical_quat,raw[:,b:b+3]-camera_position)

    def camera_visibility(self, raw):
        ball = self.camera_ball(raw)
        fov = raw.new_tensor(self.layout['camera']['fov'])*torch.pi/360
        return ((ball[:,2]>0) & (torch.atan2(ball[:,0],ball[:,2]).abs()<fov[0])
                & (torch.atan2(ball[:,1],ball[:,2]).abs()<fov[1]))

    def project_task_context(self, current, predicted):
        """Advance the same football timers and approach windows as real physics."""
        l, n = self.layout, predicted.clone()
        cfg, p, dt = l['football_feedback'], l['progress_index'], l['dt']
        q, r, b, j = l['root_quat'], l['root_xy'], l['ball_start'], l['joints_start']
        root_quat = n[:,q:q+4][:,[1,2,3,0]]
        frames = self.foot_kinematics(n[:,j:j+22])
        gaps = []
        low, high = n.new_tensor(l['foot_collision_bounds']).unbind(0)
        for position, rotation in frames.values():
            foot_position = n[:,r:r+3] + quat_rotate(root_quat, position)
            foot_quat = quat_mul(root_quat, rotation)
            local = quat_rotate_inverse(foot_quat, n[:,b:b+3]-foot_position)
            gaps.append((local-local.clamp(min=low,max=high)).norm(dim=-1)-cfg['ball_radius'])
        gap = torch.stack(gaps,-1).clamp_min(0).amin(-1)
        near = gap < cfg['near_ball_gap'] + current[:,p+3]*cfg['near_ball_hysteresis']
        goal_delta = n.new_tensor([7.,0.])-current[:,b:b+2]
        goal_direction = goal_delta/goal_delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        speed = ((n[:,b:b+2]-current[:,b:b+2])*goal_direction).sum(-1)/dt
        near_idle = torch.where(near & (speed <= cfg['ball_progress_speed']),current[:,p+1]+dt,0.)
        steps = (current[:,p+6]/dt).round().long()
        anchor_base = torch.where((steps==0)[:,None],n[:,r:r+2],current[:,p+7:p+9])
        anchor_ball = torch.where((steps==0)[:,None],n[:,b:b+2],current[:,p+9:p+11])
        window_steps = max(1,round(cfg['approach_window_s']/dt))
        steps = steps+1
        finished = steps >= window_steps
        net_approach = (anchor_base-anchor_ball).norm(dim=-1)-(n[:,r:r+2]-anchor_ball).norm(dim=-1)
        productive = net_approach >= cfg['approach_progress_speed']*cfg['approach_window_s']
        far_idle = torch.where(near | (finished & productive),0.,
                              current[:,p+2]+(finished & ~near).float()*window_steps*dt)
        pending = torch.where(finished,0,steps).float()*dt
        n[:,p+1], n[:,p+2], n[:,p+3] = near_idle, far_idle, near.float()
        n[:,p+4] = ((near_idle-cfg['near_ball_grace_s'])/cfg['near_ball_stall_ramp_s']).clamp(0,1)
        n[:,p+5] = torch.where(near,0.,((far_idle+pending-cfg['far_ball_grace_s'])/cfg['far_ball_stall_ramp_s']).clamp(0,1))
        n[:,p+6], n[:,p+7:p+9], n[:,p+9:p+11] = pending, anchor_base, anchor_ball
        return n

    @staticmethod
    def optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite learning loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()

    def fit_model(self, batch):
        states, actions, targets, rewards, dones, falls = batch
        targets = self.normalize(targets)
        prediction = self.normalize(states[:, 0])
        alive = torch.ones_like(rewards[:, 0])
        total = prediction.new_zeros(())
        mass = prediction.new_zeros(())
        for t in range(actions.shape[1]):
            prediction, reward, terminal = self.predict(prediction, actions[:, t])
            error = (prediction - targets[:, t]).square()
            if self.model_indices.numel():
                # Sensor noise and history are not physical prediction targets.
                l = self.layout
                v, end = l['velocity_start'], l['velocity_end']
                loss = error[:,l['physical_start']:v].mean(-1)
                velocity_error = (prediction[:,v:end]-targets[:,t,v:end])*self.scale[v:end]/self.velocity_delta_scale
                # Root/control accelerations must not be hidden by the much larger
                # range of velocities accumulated over an entire falling episode.
                velocity_loss = F.smooth_l1_loss(velocity_error, torch.zeros_like(velocity_error), reduction='none')
                root = l['root_velocity']-v
                loss = loss + velocity_loss.mean(-1) + velocity_loss[:,root:root+6].mean(-1)
            else:
                loss = error.mean(-1)
                if self.layout is not None:
                    loss = loss + error[:, self.layout['physical_start']:self.layout['physical_end']].mean(-1)
            loss = loss + F.smooth_l1_loss(reward, rewards[:, t] * .1, reduction='none')
            loss = loss + F.binary_cross_entropy_with_logits(terminal, falls[:, t], reduction='none')
            total = total + (alive * loss).sum()
            mass = mass + alive.sum()
            # Never fit across either a timeout or a physical termination/reset.
            alive = alive * (1 - dones[:, t])
        loss = total / mass.clamp_min(1)
        self.optimize(loss, self.model_optimizer,
                      list(self.dynamics.parameters()) + list(self.outcomes.parameters()))
        return loss.item()

    def configure_task(self, layout):
        self.layout = layout
        if 'foot_kinematics' in layout:
            from utils.kinematics import ResetKinematics
            fk = layout['foot_kinematics']
            self.foot_kinematics = ResetKinematics(fk['urdf'], fk['dof_names'], fk['frames'], self.mean.device)
            if 'camera' in layout:
                self.camera_kinematics = ResetKinematics(fk['urdf'],fk['dof_names'],[layout['camera']['frame']],self.mean.device)
        self.value_domain_positions = [i for i,index in enumerate(layout.get('model_feature_indices', []))
                                       if 262 <= index < 308 or index >= 390]
        if layout.get('model_feature_indices') and not self.model_indices.numel():
            if self.model_optimizer is not None:
                raise RuntimeError('Configure physical model before creating optimizers')
            device = self.mean.device
            self.model_indices = torch.tensor(layout['model_feature_indices'], dtype=torch.long, device=device)
            size = len(layout['model_feature_indices'])
            self.model_mean = torch.zeros(size, device=device)
            self.model_scale = torch.ones(size, device=device)
            self.velocity_delta_scale = torch.ones(layout['velocity_end']-layout['velocity_start'], device=device)
            self.dynamics = mlp(size+22, (self.model_width,)*3, layout['velocity_end']-layout['velocity_start']).to(device)
            self.outcomes = mlp(size+22+layout['physical_end']-layout['physical_start']+18, (self.model_width,)*2, 2).to(device)
            nn.init.normal_(self.dynamics[-1].weight, std=.001)
            nn.init.zeros_(self.dynamics[-1].bias)
            value_dim = self.value_features(self.mean.new_zeros(1,self.dim)).shape[-1]
            self.critic = mlp(value_dim, (self.policy_width,)*2, 1).to(device)
            nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)
        self.layout = layout
        # Keep domain parameters and curriculum context fixed in model rollouts.
        self.dynamic_mask.fill_(0)
        for start, end in layout['dynamic_ranges']:
            self.dynamic_mask[start:end] = 1

    @torch.no_grad()
    def validate_model(self, batch):
        states, actions, targets, _, dones, _ = batch
        if actions.shape[1] > self.validation_horizon:
            raise ValueError('Validation sequence exceeds configured horizon')
        initial = self.normalize(states[:, 0])
        predicted = initial
        total = initial.new_zeros(5)
        mass = initial.new_zeros(())
        alive = initial.new_ones(len(initial))
        a, b = (self.layout['physical_start'], self.layout['physical_end']) if self.layout else (0, self.dim)
        for t in range(actions.shape[1]):
            predicted = self.predict(predicted, actions[:, t])[0]
            target = self.normalize(targets[:, t])
            error = (predicted[:, a:b]-target[:, a:b]).square().mean(-1)
            persistence = (initial[:, a:b]-target[:, a:b]).square().mean(-1)
            if self.layout:
                raw = predicted*self.scale+self.mean
                l = self.layout
                j, v = l['joints_start'], l['root_velocity']
                height = (raw[:, l['root_z']]-targets[:, t, l['root_z']]).abs()
                joints = (raw[:, j:j+22]-targets[:, t, j:j+22]).square().mean(-1)
                velocity = (raw[:, v:v+6]-targets[:, t, v:v+6]).square().mean(-1)
            else:
                height = joints = velocity = error
            total += torch.stack([(x*alive).sum() for x in (error, persistence, height, joints, velocity)])
            mass += alive.sum()
            alive *= 1-dones[:, t]
        values = total/mass.clamp_min(1)
        return {'model_validation/physical_mse': values[0].item(),
                'model_validation/persistence_ratio': (values[0]/values[1].clamp_min(1e-6)).item(),
                'model_validation/height_mae_m': values[2].item(),
                'model_validation/joint_rmse_rad': values[3].sqrt().item(),
                'model_validation/root_velocity_rmse': values[4].sqrt().item()}

    def fit_real_values(self, rollout, gamma=.995, lam=.98, epochs=4):
        states, _, targets, rewards, dones, terminated = rollout
        with torch.no_grad():
            flat = states.flatten(0,1)
            # Long visual histories need not be normalized in one giant tensor.
            values = torch.cat([self.value(self.normalize(part)).squeeze(-1)
                                for part in flat.split(2048)]).reshape(rewards.shape)
            next_values = torch.cat([self.value(self.normalize(part)).squeeze(-1)
                                     for part in targets.flatten(0,1).split(2048)]).reshape(rewards.shape)
            _, returns = real_gae(rewards * .1, values, next_values, dones, terminated, gamma, lam)
            ids = torch.randperm(len(flat), device=flat.device)[:8192]
            x, target = self.normalize(flat[ids]), returns.flatten()[ids]
            before = (self.value(x).squeeze(-1) - target).square().mean()
        losses = []
        for _ in range(epochs):
            for batch in torch.randperm(len(x), device=x.device).split(2048):
                loss = F.mse_loss(self.value(x[batch]).squeeze(-1), target[batch])
                self.optimize(loss, self.critic_optimizer, self.critic.parameters())
                losses.append(loss.item())
        with torch.no_grad():
            error = self.value(x).squeeze(-1) - target
            self.last_diagnostics = {'learning/value_mse_before': before.item(),
                                     'learning/value_mse_after': error.square().mean().item()}
        return sum(losses) / len(losses)

    def task_reward(self, current, following, action):
        """Known smooth soccer objective, never the learned reward head.

        Dense terms match task units. Discrete contact/goal bonuses remain in
        real critic targets and real-simulator evaluation, not invented gradients.
        """
        c, n = current*self.scale+self.mean, following*self.scale+self.mean
        l, dt = self.layout, self.layout['dt']
        q = l['root_quat']
        w, x, y, z = n[:, q:q+4].unbind(-1)
        upright = (2*(x*z-w*y)).square() + (2*(y*z+w*x)).square()
        height, v = n[:, l['root_z']], l['root_velocity']
        speed2 = n[:, v:v+6].square().sum(-1)
        continuation = torch.sigmoid((height-.2)/.015)*torch.sigmoid((50-speed2)/3)
        progress = c[:, l['progress_index']].clamp(0, 1)
        stall = torch.maximum(n[:, l['progress_index']+4], n[:, l['progress_index']+5]).clamp(0, 1)
        reward = (3*(1-progress*stall) - 50*(height-l['nominal_height']).square() - 5*upright)*dt
        reward = reward - dt*(action-c[:, 57:79]).square().sum(-1)
        # MuJoCo free-joint angular velocity is body-local; task root_acc uses world axes.
        old_angular = quat_rotate(c[:,q:q+4][:,[1,2,3,0]], c[:,v+3:v+6])
        new_angular = quat_rotate(n[:,q:q+4][:,[1,2,3,0]], n[:,v+3:v+6])
        acceleration_delta = torch.cat((n[:,v:v+3]-c[:,v:v+3], new_angular-old_angular), -1)
        reward = reward - .001/dt*acceleration_delta.square().sum(-1)
        j, d = l['joints_start'], l['default_start']
        reward = reward - dt*(1-progress)*(n[:, j:j+22]-n[:, d:d+22]).square().mean(-1)
        root, ball = l['root_xy'], l['ball_start']
        bp, bn = c[:, ball:ball+2], n[:, ball:ball+2]
        approach = (bn-c[:, root:root+2]).norm(dim=-1) - (bn-n[:, root:root+2]).norm(dim=-1)
        goal = bn.new_tensor([7., 0.])
        forward = (bp-goal).norm(dim=-1) - (bn-goal).norm(dim=-1)
        gain = 1 + c[:, l['group_index']+2].clamp(0, 1)*(l['near_approach_gain']-1)
        reward = reward + progress*dt*(50*gain*approach + 500*forward)
        cost = 2*n[:, l['progress_index']+4] + n[:, l['progress_index']+5]
        reward = reward - progress*dt*l['stall_penalty_factor']*cost
        if 'camera' in l:
            camera_ball = self.camera_ball(n)
            z = camera_ball[:,2]
            z = torch.where(z.abs()<1e-8,z.sign()*1e-8,z)
            reward = reward+progress*dt*(
                l['camera']['pitch_reward_scale']*torch.atan2(camera_ball[:,1],z).square()
                +l['camera']['yaw_reward_scale']*torch.atan2(camera_ball[:,0],z).square())
        return .1*(reward-20*(1-continuation)), continuation

    def imagined_objective(self, initial, horizon, gamma=.995, noise=None, visual_noise=None):
        state = self.normalize(initial).detach()
        alive, total = state.new_ones(len(state)), state.new_zeros(len(state))
        for t in range(horizon):
            mean = self.actor(self.policy_observation(state))
            action = self.action_limit * (mean if noise is None else mean + noise[t]).tanh()
            following, _, _ = self.predict(state, action, visual_noise=None if visual_noise is None else visual_noise[t])
            reward, continuation = self.task_reward(state, following, action)
            total = total + gamma**t * alive * reward
            alive = alive * continuation
            state = following
        return (total + gamma**horizon*alive*self.value(state).squeeze(-1)).mean()

    def improve_model_policy(self, initial, horizon, exploration_std, max_kl=.01, gamma=.995, reference_state=None):
        import copy
        parameters = list(self.actor.parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
        obs = self.policy_observation(self.normalize(initial)).detach()
        with torch.no_grad():
            old_mean = self.actor(obs)
            reference_mean = (old_mean if reference_state is None else
                              torch.func.functional_call(self.actor, reference_state, (obs,)))
        noise = torch.randn(horizon, len(initial), 22, device=initial.device) * (exploration_std/float(self.action_limit))
        visual_noise = torch.cat((torch.randn(horizon,len(initial),2,device=initial.device),
                                  torch.rand(horizon,len(initial),1,device=initial.device)),-1)
        with frozen(self.dynamics, self.outcomes, self.critic):
            objective = self.imagined_objective(initial, horizon, gamma=gamma, noise=noise, visual_noise=visual_noise)
            loss = -objective
            self.optimize(loss, self.actor_optimizer, parameters)
            proposed = [p.detach().clone() for p in parameters]
            alpha, kl = 1., 0.
            with torch.no_grad():
                for _ in range(12):
                    for p, old, new in zip(parameters, before, proposed):
                        p.copy_(old + alpha*(new-old))
                    updated_mean = self.actor(obs)
                    variance = 2*(exploration_std/float(self.action_limit))**2
                    kl = max(((updated_mean-anchor).square()/variance).sum(-1).mean().item()
                             for anchor in (old_mean, reference_mean))
                    improved = self.imagined_objective(initial, horizon, gamma=gamma, noise=noise, visual_noise=visual_noise)
                    if kl <= max_kl and improved >= objective.detach()-1e-7:
                        break
                    alpha *= .5
                else:
                    for p, old in zip(parameters, before):p.copy_(old)
                    alpha = 0.
                    kl = ((old_mean-reference_mean).square() /
                          (2*(exploration_std/float(self.action_limit))**2)).sum(-1).mean().item()
                if alpha < 1:self.actor_optimizer.load_state_dict(optimizer_before)
        self.last_diagnostics.update({'learning/policy_kl': kl, 'learning/update_fraction': alpha})
        return loss.item()


def real_gae(rewards, values, next_values, dones, terminated, gamma=.995, lam=.98):
    """Separate bootstrap and trace masks; never propagate through auto-reset."""
    advantages = torch.zeros_like(rewards)
    future = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1-terminated[t]) * next_values[t] - values[t]
        future = delta + gamma * lam * (1-dones[t]) * future
        advantages[t] = future
    return advantages.detach(), (advantages + values).detach()
