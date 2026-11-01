"""Batched GPU soccer. Physics, observations, rewards and resets stay on CUDA."""
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from .env import SoccerEnv
from .curriculum import STAGES, AMPLITUDES
from .warp_config import CollisionCapacity


@wp.kernel
def pd(q: wp.array2d[float], v: wp.array2d[float], qi: wp.array[int], vi: wp.array[int],
       kp: wp.array[float], kd: wp.array[float], target: wp.array2d[float],
       limits: wp.array2d[float], ctrl: wp.array2d[float]):
    w, j = wp.tid()
    torque = kp[j % 22]*(target[w, j]-q[w, qi[j]])-kd[j % 22]*v[w, vi[j]]
    ctrl[w, j] = wp.clamp(torque, limits[j, 0], limits[j, 1])


@wp.kernel
def collect_touches(nacon: wp.array[int], geom: wp.array[wp.vec2i], world: wp.array[int],
                    dist: wp.array[float], owner: wp.array[int], ball: int,
                    touches: wp.array2d[int]):
    c = wp.tid()
    if c < nacon[0] and dist[c] <= 0.0:
        a, b = geom[c][0], geom[c][1]
        other = int(-1)
        if a == ball:
            other = b
        elif b == ball:
            other = a
        if other >= 0:
            robot = owner[other]
            if robot >= 0:
                wp.atomic_max(touches, world[c], robot, 1)


@wp.kernel
def monitor(q: wp.array2d[float], v: wp.array2d[float], overflow: wp.array[int], failed: wp.array[int]):
    w, j = wp.tid()
    if not wp.isfinite(q[w, j]):
        wp.atomic_or(failed, w, 2)
    if j < v.shape[1]:
        if not wp.isfinite(v[w, j]):
            wp.atomic_or(failed, w, 2)
    if j == 0 and overflow[w] != 0:
        wp.atomic_or(failed, w, (overflow[w] << 2) | 1)


class WarpSoccerEnv:
    dt = .02
    num_agents = 8

    def __init__(self, count=64, device='cuda:0', seed=42, stage='balance', episode_steps=300,
                 nconmax=768, njmax=3072):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Warp backend requires CUDA; use --backend cpu for reference tests')
        self.count, self.stage, self.episode_steps = count, stage, episode_steps
        self.reference = ref = SoccerEnv(seed=seed, stage=stage)
        self.model = ref.model
        self.physics_config=CollisionCapacity.configure(self.model.opt.ccd_iterations)
        self.observation_space, self.share_observation_space, self.action_space = ref.observation_space, ref.share_observation_space, ref.action_space
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.wp_device = str(self.device)
        wp.init()
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        torch.cuda.set_stream(self.torch_stream)
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        for name in ('qidx', 'vidx', 'rootq', 'rootv', 'bodies'):
            setattr(self, name, self.tensor(getattr(ref, name), torch.long))
        for name in ('neutral', 'scale', 'limits', 'kp', 'kd', 'sign', 'roles'):
            setattr(self, name, self.tensor(getattr(ref, name)))
        self.ballq, self.ballv = ref.ballq, ref.ballv
        with wp.ScopedStream(self.stream):
            self.wmodel = mjw.put_model(self.model)
            self.data = mjw.make_data(self.model, nworld=count, nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xmat = wp.to_torch(self.data.xmat)
        self.target = torch.zeros((count, 176), device=self.device)
        self.touches = torch.zeros((count, 8), device=self.device, dtype=torch.int32)
        self.steps = torch.zeros(count, device=self.device, dtype=torch.long)
        self.practice = torch.zeros(count, device=self.device, dtype=torch.long)
        self.previous_action = torch.zeros((count, 8, 22), device=self.device)
        self.start_positions = torch.zeros((count, 8, 3), device=self.device)
        self.targets = torch.zeros_like(self.start_positions)
        self.start_ball = torch.zeros((count, 3), device=self.device)
        self.ever_touched = torch.zeros((count, 8), device=self.device, dtype=torch.bool)
        self.reached = torch.zeros_like(self.ever_touched)
        self.best_progress = torch.zeros(count, device=self.device)
        self.best_approach = torch.full((count,), torch.inf, device=self.device)
        self.rows = torch.arange(count, device=self.device)
        self.others = self.tensor([[j for j in range(i//4*4, i//4*4+4) if j != i]+
                                  list(range((1-i//4)*4, (1-i//4)*4+4)) for i in range(8)], torch.long)
        owner = np.full(self.model.ngeom, -1, dtype=np.int32)
        for g, i in ref.geom_owner.items():
            owner[g] = i
        self._wp_arrays = [wp.array(ref.qidx.ravel().astype(np.int32), device=self.wp_device),
                           wp.array(ref.vidx.ravel().astype(np.int32), device=self.wp_device),
                           wp.from_torch(self.kp), wp.from_torch(self.kd),
                           wp.from_torch(self.target), wp.array(self.model.actuator_ctrlrange.astype(np.float32), device=self.wp_device)]
        self.owner_wp = wp.array(owner, device=self.wp_device)
        self.failure = wp.zeros(count, dtype=wp.int32, device=self.wp_device)
        self.touches_wp = wp.from_torch(self.touches)
        # Compile outside capture, then restore state. Graph warmup never advances an episode.
        with wp.ScopedStream(self.stream):
            self._substep()
            mjw.forward(self.wmodel, self.data)
            with wp.ScopedCapture(stream=self.stream) as cap:
                for _ in range(10):
                    self._substep()
                mjw.forward(self.wmodel, self.data)
            self.graph = cap.graph
            with wp.ScopedCapture(stream=self.stream) as cap:
                mjw.forward(self.wmodel, self.data)
            self.forward_graph = cap.graph
        self.failure.zero_()
        self.reset()

    def _substep(self):
        wp.launch(pd, dim=(self.count, 176), inputs=[self.data.qpos, self.data.qvel, *self._wp_arrays, self.data.ctrl], device=self.wp_device)
        mjw.step(self.wmodel, self.data)
        wp.launch(monitor, dim=(self.count, self.model.nq),
                  inputs=[self.data.qpos, self.data.qvel, self.data.overflow, self.failure], device=self.wp_device)
        wp.launch(collect_touches, dim=self.data.contact.geom.shape[0],
                  inputs=[self.data.nacon, self.data.contact.geom, self.data.contact.worldid,
                          self.data.contact.dist, self.owner_wp, self.reference.ballgeom, self.touches_wp], device=self.wp_device)

    def uniform(self, shape, low=0., high=1.):
        return torch.rand(shape, generator=self.generator, device=self.device)*(high-low)+low

    @property
    def ball(self):
        return self.qpos[:, self.ballq:self.ballq+3]

    def reset(self, ids=None):
        ids = self.rows if ids is None else ids
        if ids.numel() == 0:
            return self.observe(), self.state()
        n = ids.numel()
        mask = torch.zeros(self.count, device=self.device, dtype=torch.bool)
        mask[ids] = True
        with wp.ScopedStream(self.stream):
            mjw.reset_data(self.wmodel, self.data, wp.from_torch(mask))
        q = self.tensor(self.model.qpos0).expand(n, -1).clone()
        formation = self.tensor([[-5.8, 0], [-3.8, 0], [-1.8, -1.5], [-1.8, 1.5]]).repeat(2, 1)*self.sign[:, None]
        q[:, self.rootq[:, None]+torch.arange(3, device=self.device)] = torch.cat([
            formation.expand(n, -1, -1)+self.uniform((n, 8, 2), -.08, .08),
            torch.full((n, 8, 1), .417, device=self.device)], dim=-1)
        quats = self.tensor([[1, 0, 0, 0]]*4+[[0, 0, 0, 1]]*4)
        q[:, self.rootq[:, None]+torch.arange(3, 7, device=self.device)] = quats
        q[:, self.qidx] = self.neutral
        q[:, self.ballq:self.ballq+2] = self.uniform((n, 2), -.25, .25)
        q[:, self.ballq+2] = .101
        q[:, self.ballq+3:self.ballq+7] = self.tensor([1, 0, 0, 0])
        practice = torch.randint(0, 2, (n,), generator=self.generator, device=self.device)*4+2
        self.practice[ids] = practice
        if self.stage in ('approach', 'kick', 'teamplay'):
            sign = self.sign[practice]
            x = sign*(4.8 if self.stage == 'kick' else 1.5)
            q[:, self.ballq] = x
            q[:, self.ballq+1] = 0
            r = torch.arange(n, device=self.device)
            q[r, self.rootq[practice]] = x-sign*{'approach': 1., 'kick': .32, 'teamplay': .65}[self.stage]
            q[r, self.rootq[practice]+1] = 0
        self.qpos[ids] = q
        self.start_positions[ids] = q[:, self.rootq[:, None]+torch.arange(3, device=self.device)]
        self.targets[ids] = self.start_positions[ids]
        self.targets[ids, :, 0] += self.sign*self.uniform((n, 8), .6, 1.)
        self.targets[ids, :, 1] += self.uniform((n, 8), -.3, .3)
        self.start_ball[ids] = self.ball[ids]
        self.steps[ids] = 0
        self.previous_action[ids] = 0
        self.ever_touched[ids] = False
        self.reached[ids] = False
        self.best_progress[ids] = 0
        self.best_approach[ids] = torch.inf
        self.touches[ids] = 0
        self.forward()
        return self.observe(), self.state()

    def forward(self):
        wp.capture_launch(self.forward_graph,stream=self.stream)

    def observe(self):
        n = self.count
        pos = self.xpos[:, self.bodies]
        rot = self.xmat[:, self.bodies].reshape(n, 8, 3, 3)
        local = lambda v: torch.einsum('nai,naij->naj', v, rot)
        velocity = self.qvel[:, self.rootv[:, None]+torch.arange(6, device=self.device)]
        goal = torch.stack((self.sign*7, torch.zeros_like(self.sign), torch.full_like(self.sign, .1)), -1)
        other = self.xpos[:, self.bodies[self.others]]-pos[:, :, None, :]
        other = torch.einsum('naki,naij->nakj', other, rot).flatten(2)/7
        stage = torch.nn.functional.one_hot(torch.tensor(STAGES.index(self.stage), device=self.device), len(STAGES)).expand(n, 8, -1)
        time = (self.steps/self.episode_steps)[:, None, None].expand(n, 8, 1)
        return torch.cat([self.qpos[:, self.qidx]-self.neutral, self.qvel[:, self.vidx]*.1,
                          rot[:, :, 2], local(velocity[:, :, :3]), velocity[:, :, 3:]*.1,
                          pos[:, :, 2:3], local(self.ball[:, None]-pos)/7,
                          local(self.qvel[:, None, self.ballv:self.ballv+3].expand(-1, 8, -1))*.2,
                          local(goal-pos)/7, other, self.previous_action, local(self.targets-pos),
                          stage, self.roles.expand(n, -1, -1), time,
                          torch.full_like(time, AMPLITUDES[STAGES.index(self.stage)])], -1)

    def state(self, obs=None):
        obs = self.observe() if obs is None else obs
        red = obs.flatten(1)[:, None].expand(-1, 4, -1)
        blue = torch.cat((obs[:, 4:], obs[:, :4]), 1).flatten(1)[:, None].expand(-1, 4, -1)
        return torch.cat((torch.cat((red, blue), 1), self.roles.expand(self.count, -1, -1)), -1)

    def check(self):
        if (wp.to_torch(self.data.overflow).any() | (wp.to_torch(self.failure) != 0).any()).item():
            from mujoco_warp._src.types import OverflowType
            codes=wp.to_torch(self.failure).unique().cpu().tolist()
            raw_codes=wp.to_torch(self.data.overflow).unique().cpu().tolist()
            bits=0
            for code in codes: bits |= int(code)>>2
            for code in raw_codes: bits |= int(code)
            names=[flag.name for flag in OverflowType if int(flag)&bits]
            raise RuntimeError(f'Warp failure latched: codes={codes}, overflow={names}; resets do not clear failures')
        if not (torch.isfinite(self.qpos).all() & torch.isfinite(self.qvel).all()).item():
            raise FloatingPointError('Nonfinite Warp state')

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.count, 8, 22) or action.device != self.device:
            raise ValueError('Actions must be CUDA tensors [num_envs,8,22] on the physics device')
        applied = torch.tanh(action)
        target = (self.neutral+getattr(self,'course_amplitude',AMPLITUDES[STAGES.index(self.stage)])*self.scale*applied).clamp(self.limits[:, 0], self.limits[:, 1])
        self.target.copy_(target.flatten(1))
        before = self.ball.clone()
        oldpos = self.xpos[:, self.bodies]
        norm = lambda x: torch.linalg.vector_norm(x, dim=-1)
        olddist = norm(oldpos[:, :, :2]-before[:, None, :2])
        oldtarget = norm(oldpos[:, :, :2]-self.targets[:, :, :2])
        self.touches.zero_()
        wp.capture_launch(self.graph,stream=self.stream)
        self.steps += 1
        pos = self.xpos[:, self.bodies]
        upright = self.xmat[:, self.bodies].reshape(self.count, 8, 3, 3)[:, :, 2, 2]
        fallen = (pos[:, :, 2] < .20) | (upright < .3)
        outside = (pos[:, :, 0].abs() > 7.7) | (pos[:, :, 1].abs() > 4.7)
        goal = torch.zeros(self.count, device=self.device)
        for sign in (1, -1):
            a, b = sign*before[:, 0], sign*self.ball[:, 0]
            fraction = (7.1-a)/(b-a).clamp(min=1e-8)
            crossing = before+fraction[:, None]*(self.ball-before)
            scored = (a <= 7.1) & (b > 7.1) & (crossing[:, 1].abs() <= 1.2) & (crossing[:, 2] >= 0) & (crossing[:, 2] <= .9)
            goal += sign*scored
        out = (self.ball[:, 1].abs() > 4.6) | ((self.ball[:, 0].abs() > 7.1) & (goal == 0))
        terminated = (goal != 0) | out | fallen.any(1) | outside.any(1)
        truncated = (self.steps >= self.episode_steps) & ~terminated
        reward = self.dt*(1-2*(1-upright)-10*(pos[:, :, 2]-.417).square()
                          -.02*self.qvel[:, self.vidx].square().mean(-1)
                          -.1*(applied-self.previous_action).square().mean(-1))
        delta = self.ball[:, 0]-before[:, 0]
        if self.stage in ('approach', 'kick', 'teamplay', 'match'):
            approach = (olddist-norm(pos[:, :, :2]-self.ball[:, None, :2])).clamp(-.05, .05)
            reward += self.sign*delta.clamp(-.2, .2)[:, None]*2
            if self.stage in ('approach', 'kick'):
                reward[self.rows, self.practice] += approach[self.rows, self.practice]*2
                not_practice = torch.arange(8, device=self.device)[None] != self.practice[:, None]
                reward -= self.dt*.1*norm(pos[:, :, :2]-self.start_positions[:, :, :2])*not_practice
            else:
                reward += approach*.2
                anchors = self.ball[:, None, :2].expand(-1, 8, -1).clone()
                anchors[:, [0, 4], 0] = -6*self.sign[[0, 4]]
                anchors[:, [0, 4], 1] = self.ball[:, 1:2].clamp(-1, 1)
                anchors[:, [1, 5], 0] = -3*self.sign[[1, 5]]
                anchors[:, [1, 5], 1] = self.ball[:, 1:2]*.5
                anchors[:, [3, 7], 0] -= self.sign[[3, 7]]
                anchors[:, [3, 7], 1] += 1.5
                reward -= self.dt*.03*norm(pos[:, :, :2]-anchors).clamp(max=8)
            reward += self.sign*goal[:, None]*20
        targetdist = norm(pos[:, :, :2]-self.targets[:, :, :2])
        if self.stage == 'move':
            reward += 3*(oldtarget-targetdist).clamp(-.05, .05)-self.dt*.2*targetdist.clamp(max=2)
        self.reached |= targetdist < .25
        self.ever_touched |= self.touches.bool()
        self.best_approach = torch.minimum(self.best_approach, norm(pos[self.rows, self.practice, :2]-self.ball[:, :2]))
        practice_touch = self.ever_touched[self.rows, self.practice]
        self.best_progress = torch.maximum(self.best_progress, torch.where(practice_touch,
            self.sign[self.practice]*(self.ball[:, 0]-self.start_ball[:, 0]), 0.))
        if self.stage == 'kick':
            reward[self.rows, self.practice] += (self.sign[self.practice]*delta).clamp(min=0)*5*self.touches[self.rows, self.practice]
        reward -= 5*(fallen | outside)
        self.previous_action.copy_(applied)
        touched_goal = (goal != 0) & self.ever_touched.any(1)
        stable = ~fallen.any(1) & ~outside.any(1) & ((self.steps*self.dt >= 5) | touched_goal)
        team = torch.arange(8, device=self.device)[None]//4 == self.practice[:, None]//4
        objectives = {'balance': torch.ones_like(stable), 'move': self.reached.float().mean(1) >= .75,
                      'approach': self.best_approach < .45, 'kick': self.best_progress >= .3,
                      'teamplay': ((self.best_progress >= 1) & ((self.ever_touched & team).sum(1) >= 2)) | touched_goal,
                      'match': touched_goal}
        info = dict(goal=goal, fallen=fallen.any(1), out=out, stable=stable,
                    success=stable & objectives[self.stage], steps=self.steps.clone(), stage=self.stage)
        obs = self.observe()
        return obs, self.state(obs), reward[..., None], terminated, truncated, info
