"""Full free-base MuJoCo soccer. No kinematic root/ball motion during steps."""
import numpy as np
import mujoco
import yaml
from gymnasium.spaces import Box
from .scene import MUJOCO_ROOT, SCENE, PREFIXES, build_scene
from .curriculum import STAGES, AMPLITUDES


class SoccerEnv:
    num_agents = 8
    dt = 0.02

    def __init__(self, seed=42, episode_steps=1500, stage='balance', model=None):
        if not SCENE.exists():
            build_scene()
        self.model = model or mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.episode_steps, self.stage = episode_steps, stage
        if stage not in STAGES:
            raise ValueError(stage)
        cfg = yaml.safe_load((MUJOCO_ROOT / 'resources/Q1.yaml').read_text())
        names = cfg['asset']['dof_names']
        self.qidx = np.array([[self.model.jnt_qposadr[self.model.joint(p+n).id] for n in names] for p in PREFIXES])
        self.vidx = np.array([[self.model.jnt_dofadr[self.model.joint(p+n).id] for n in names] for p in PREFIXES])
        self.bodies = np.array([self.model.body(p+'pelvis').id for p in PREFIXES])
        self.rootq = np.array([self.model.jnt_qposadr[self.model.joint(p+'floating_base_joint').id] for p in PREFIXES])
        self.rootv = np.array([self.model.jnt_dofadr[self.model.joint(p+'floating_base_joint').id] for p in PREFIXES])
        self.ballq = self.model.jnt_qposadr[self.model.joint('ball_free').id]
        self.ballv = self.model.jnt_dofadr[self.model.joint('ball_free').id]
        self.ballgeom = self.model.geom('soccer_ball').id
        self.geom_owner = {}
        for g in range(self.model.ngeom):
            b = self.model.geom_bodyid[g]
            name = self.model.body(b).name or ''
            for i, p in enumerate(PREFIXES):
                if name.startswith(p):
                    self.geom_owner[g] = i
        self.neutral = np.array([cfg['init_state']['default_joint_angles'].get(n, 0.) for n in names])
        self.kp, self.kd = [np.array([next(v for k, v in cfg['control'][key].items() if k in n) for n in names]) for key in ('stiffness', 'damping')]
        self.scale = np.array(cfg['control']['action_scale'])
        self.limits = np.array([self.model.jnt_range[self.model.joint(PREFIXES[0]+n).id] for n in names])
        self.sign = np.array([1.]*4 + [-1.]*4)
        self.roles = np.tile(np.eye(4), (2, 1))
        self.reset()
        self.observation_space = Box(-np.inf, np.inf, self.observe().shape[1:], dtype=np.float32)
        self.share_observation_space = Box(-np.inf, np.inf, (self.observe().size + 4,), dtype=np.float32)
        self.action_space = Box(-np.inf, np.inf, (22,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        formation = np.array([[-5.8, 0], [-3.8, 0], [-1.8, -1.5], [-1.8, 1.5]])
        for i, q in enumerate(self.rootq):
            xy = formation[i % 4] * self.sign[i] + self.rng.uniform(-.08, .08, 2)
            self.data.qpos[q:q+3] = [*xy, .417]
            self.data.qpos[q+3:q+7] = [1, 0, 0, 0] if i < 4 else [0, 0, 0, 1]
        self.data.qpos[self.qidx] = self.neutral
        self.data.qpos[self.ballq:self.ballq+3] = [*self.rng.uniform(-.25, .25, 2), .101]
        self.data.qpos[self.ballq+3:self.ballq+7] = [1, 0, 0, 0]
        # Practice both attack directions, with one designated striker near the ball.
        self.practice = int(self.rng.choice([2, 6]))
        sign = self.sign[self.practice]
        if self.stage in ('approach', 'kick', 'teamplay'):
            x = sign * (4.8 if self.stage == 'kick' else 1.5)
            self.data.qpos[self.ballq:self.ballq+3] = [x, 0, .101]
            gap = {'approach': 1.0, 'kick': .32, 'teamplay': .65}[self.stage]
            q = self.rootq[self.practice]
            self.data.qpos[q:q+2] = [x-sign*gap, 0]
        self.start_positions = self.data.qpos[self.rootq[:, None] + np.arange(3)].copy()
        self.targets = self.start_positions.copy()
        self.targets[:, 0] += self.sign * self.rng.uniform(.6, 1., 8)
        self.targets[:, 1] += self.rng.uniform(-.3, .3, 8)
        self.start_ball = self.ball.copy()
        self.ever_touched = set()
        self.best_progress = 0.
        self.best_approach = np.inf
        self.reached = np.zeros(8, dtype=bool)
        self.previous_action = np.zeros((8, 22))
        self.steps = 0
        self.last_touch = -1
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), self.state()

    @property
    def ball(self):
        return self.data.qpos[self.ballq:self.ballq+3].copy()

    def observe(self):
        d = self.data
        pos = d.xpos[self.bodies]
        result = []
        for i in range(8):
            rotation = d.xmat[self.bodies[i]].reshape(3, 3)
            s = self.sign[i]
            teammates = [j for j in range(i//4*4, i//4*4+4) if j != i]
            opponents = list(range((1-i//4)*4, (1-i//4)*4+4))
            others = np.concatenate([(pos[j]-pos[i]) @ rotation for j in teammates+opponents])
            velocity = d.qvel[self.rootv[i]:self.rootv[i]+6]
            goal = (np.array([7*s, 0, .1])-pos[i]) @ rotation
            # Local proprioception + ideal relative tracking (not camera pixels).
            result.append(np.concatenate([
                d.qpos[self.qidx[i]]-self.neutral, d.qvel[self.vidx[i]]*.1,
                rotation[2], velocity[:3] @ rotation, velocity[3:]*.1,
                [pos[i, 2]], (self.ball-pos[i]) @ rotation / 7,
                d.qvel[self.ballv:self.ballv+3] @ rotation *.2, goal/7,
                others/7, self.previous_action[i],
                (self.targets[i]-pos[i]) @ rotation,
                np.eye(len(STAGES))[STAGES.index(self.stage)],
                self.roles[i], [self.steps/self.episode_steps, AMPLITUDES[STAGES.index(self.stage)]]]))
        return np.asarray(result, dtype=np.float32)

    def state(self):
        obs = self.observe()
        # Both teams' proprioception available only to centralized critic.
        return np.array([np.r_[obs[np.r_[np.arange(i//4*4, i//4*4+4), np.arange((1-i//4)*4, (1-i//4)*4+4)]].ravel(), self.roles[i]] for i in range(8)], dtype=np.float32)

    @staticmethod
    def goal_crossing(previous, current):
        # Whole ball must cross line; interpolate crossing to reject side entry.
        for sign in (1, -1):
            a, b = sign*previous[0], sign*current[0]
            if a <= 7.1 < b:
                t = (7.1-a)/(b-a)
                crossing = previous + t*(current-previous)
                if abs(crossing[1]) <= 1.2 and .0 <= crossing[2] <= .9:
                    return sign
        return 0

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (8, 22) or not np.isfinite(action).all():
            raise ValueError('Expected finite action shape (8,22)')
        applied = np.tanh(action)
        amplitude = AMPLITUDES[STAGES.index(self.stage)]
        target = np.clip(self.neutral + amplitude*self.scale*applied, self.limits[:, 0], self.limits[:, 1])
        before = self.ball
        old_target_dist = np.linalg.norm(self.data.xpos[self.bodies, :2]-self.targets[:, :2], axis=1)
        olddist = np.linalg.norm(self.data.xpos[self.bodies, :2]-before[:2], axis=1)
        touched = set()
        for _ in range(10):
            torque = self.kp*(target-self.data.qpos[self.qidx])-self.kd*self.data.qvel[self.vidx]
            self.data.ctrl[:] = np.clip(torque.ravel(), self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
            for c in self.data.contact:
                if self.ballgeom in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == self.ballgeom else c.geom1
                    if other in self.geom_owner:
                        self.last_touch = self.geom_owner[other]
                        touched.add(self.last_touch)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('MuJoCo produced nonfinite state')
        if any(w.number for w in self.data.warning):
            raise RuntimeError('MuJoCo simulation warning; refusing corrupted rollout')
        pos = self.data.xpos[self.bodies]
        upright = self.data.xmat[self.bodies, 8]
        fallen = (pos[:, 2] < .20) | (upright < .3)
        outside = (np.abs(pos[:, 0]) > 7.7) | (np.abs(pos[:, 1]) > 4.7)
        goal = self.goal_crossing(before, self.ball)
        out = abs(self.ball[1]) > 4.6 or (abs(self.ball[0]) > 7.1 and not goal)
        # A fall ends the rally for all 8 agents. Falling never pays an opponent-loss reward.
        terminated = bool(goal or out or fallen.any() or outside.any())
        truncated = self.steps >= self.episode_steps and not terminated
        reward = self.dt*(1 - 2*(1-upright) - 10*(pos[:, 2]-.417)**2
                          - .02*np.mean(self.data.qvel[self.vidx]**2, axis=1)
                          - .1*np.mean((applied-self.previous_action)**2, axis=1))
        if self.stage in ('approach', 'kick', 'teamplay', 'match'):
            newdist = np.linalg.norm(pos[:, :2]-self.ball[:2], axis=1)
            approach = np.clip(olddist-newdist, -.05, .05)
            # Shared zero-sum ball progress; role anchors discourage eight-player swarming.
            reward += self.sign*np.clip(self.ball[0]-before[0], -.2, .2)*2 + approach*.2
            if self.stage in ('approach', 'kick'):
                # Only the designated practice striker approaches; teammates maintain formation.
                reward -= approach*.2
                reward[self.practice] += approach[self.practice]*2
                reward -= self.dt*.1*np.linalg.norm(pos[:, :2]-self.start_positions[:, :2], axis=1) * (np.arange(8) != self.practice)
            for i in range(8):
                role = i % 4
                anchor = [np.array([-6*self.sign[i], np.clip(self.ball[1], -1, 1)]),
                          np.array([-3*self.sign[i], self.ball[1]*.5]),
                          self.ball[:2], self.ball[:2]+np.array([-self.sign[i], 1.5])][role]
                if self.stage in ('teamplay', 'match'):
                    reward[i] -= self.dt*.03*min(np.linalg.norm(pos[i, :2]-anchor), 8)
            reward += self.sign*goal*20
        target_dist = np.linalg.norm(pos[:, :2]-self.targets[:, :2], axis=1)
        if self.stage == 'move':
            reward += 3*np.clip(old_target_dist-target_dist, -.05, .05)
            reward -= self.dt*.2*np.minimum(target_dist, 2)
        self.reached |= target_dist < .25
        self.ever_touched.update(touched)
        self.best_approach = min(self.best_approach, np.linalg.norm(pos[self.practice, :2]-self.ball[:2]))
        if self.practice in self.ever_touched:
            self.best_progress = max(self.best_progress, self.sign[self.practice]*(self.ball[0]-self.start_ball[0]))
        if self.stage == 'kick' and self.practice in touched:
            reward[self.practice] += max(0., self.sign[self.practice]*(self.ball[0]-before[0]))*5
        reward[fallen | outside] -= 5
        self.previous_action = applied
        info = dict(goal=int(goal), out=bool(out), fallen=fallen.tolist(), touches=sorted(touched),
                    last_touch=int(self.last_touch), steps=self.steps, stage=self.stage,
                    terminated=terminated, truncated=bool(truncated))
        # A curriculum success requires all robots to remain upright for >=5 seconds.
        # Goal termination is also accepted if a real contact preceded the goal.
        stable = not fallen.any() and not outside.any() and (self.steps*self.dt >= 5 or bool(goal and self.ever_touched))
        objective = {'balance': True, 'move': self.reached.mean() >= .75,
                     'approach': self.best_approach < .45,
                     'kick': self.best_progress >= .3,
                     'teamplay': (self.best_progress >= 1. and sum(j//4 == self.practice//4 for j in self.ever_touched) >= 2) or bool(goal and self.ever_touched),
                     'match': bool(goal and self.ever_touched)}[self.stage]
        info.update(stable=bool(stable), success=bool(stable and objective),
                    ball_progress=float(self.best_progress), reached_fraction=float(self.reached.mean()))
        return self.observe(), self.state(), reward[:, None].astype(np.float32), terminated, bool(truncated), info
