"""Q1 soccer on MuJoCo Warp GPU physics with a native CPU reference.

Uses upstream soccer rewards, observations and AMP layouts. Each simulation
has its own complete field; no Isaac Gym dependency is needed for this task.
"""
import numpy as np
import mujoco
from utils.physics import Physics
import torch

from .t1 import T1
from utils.motion import Motion
from utils.kinematics import ResetKinematics
from utils.torch_math import quat_rotate, quat_rotate_inverse, quat_mul, quat_from_euler_xyz
from utils.utils import apply_randomization


MIRROR_PERMUTATION = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
                      12, 13, 18, 19, 20, 21, 14, 15, 16, 17]
MIRROR_SIGNS = [1, -1, -1, 1, 1, -1] * 2 + [-1, -1] + [1, -1, -1, 1] * 2


def mirror_joints(values):
    """Accept numpy arrays or torch tensors, including stacked observations."""
    return values[..., MIRROR_PERMUTATION] * (
        values.new_tensor(MIRROR_SIGNS) if hasattr(values, 'new_tensor') else MIRROR_SIGNS
    )


def mirror_observations(obs):
    mirrored = obs.clone() if hasattr(obs, 'clone') else obs.copy()
    mirrored[..., [1, 3, 5, 7, 10, 12]] *= -1
    for start in (13, 35, 57):
        mirrored[..., start:start + 22] = mirror_joints(obs[..., start:start + 22])
    return mirrored


class FlatTerrain:
    border_size = 20.
    env_length = 20.
    env_width = 20.

    def terrain_heights(self, xy):
        return torch.zeros(xy.shape[:-1], dtype=xy.dtype, device=xy.device)


class Q1(T1):
    mirror_act = staticmethod(mirror_joints)
    mirror_obs = staticmethod(mirror_observations)

    def __init__(self, cfg):
        self.cfg = cfg
        self.balance_phase = False
        self.balance_release_iteration = None
        self.soccer_progress = 0.
        self.reference_init_probability = 1.
        self.recent_success_rate = 0.
        self._curriculum_iteration = None
        self._curriculum_last_adjustment = None
        self._curriculum_quality_window = []
        self.amp_reward_scale = 1.
        self.near_start_progress = 0.
        self.near_start_success_rate = 0.
        self.near_start_trials = 0
        self._near_start_success_mass = 0.
        self._near_start_trial_mass = 0.
        self._near_start_last_adjustment = None
        self._near_start_last_trial_count = 0
        self.full_start_success_rate = 0.
        self._full_start_success_mass = 0.
        self._full_start_trial_mass = 0.
        self.device = cfg['basic']['device']
        self.num_envs = cfg['env']['num_envs']
        mixture = cfg['init_state'].get('spawn_mixture')
        self.spawn_group = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        if mixture is not None:
            if len(mixture) != 3 or min(mixture) < 0 or not np.isclose(sum(mixture), 1.):
                raise ValueError('spawn_mixture must contain three nonnegative fractions summing to one')
            boundaries = torch.tensor(np.cumsum(mixture)[:2], device=self.device)
            quantiles = (torch.arange(self.num_envs, device=self.device) + .5) / self.num_envs
            self.spawn_group[:] = torch.bucketize(quantiles, boundaries)
        self.num_dofs = cfg['env']['num_actions']
        self.dof_names = cfg['asset']['dof_names']
        self.terrain = FlatTerrain()
        self.up_axis_idx = 2
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
        self.model.opt.timestep = cfg['sim']['dt']
        if [self.model.actuator(i).name for i in range(self.model.nu)] != self.dof_names:
            raise ValueError('MuJoCo actuator order must match the Q1 policy')
        threads = cfg.get('physics', {}).get('num_threads', 8)
        self.batch = Physics(self.model, self.num_envs, self.device, threads)
        self.cfg['sim']['physics_engine'] = 'mujoco_warp' if self.batch.gpu else 'mujoco_cpu'
        self.qpos = self.batch.bind('qpos')
        self.qvel = self.batch.bind('qvel')
        self.ctrl = self.batch.bind('ctrl')
        self.xfrc = self.batch.bind('xfrc_applied')
        self.warmstart = self.batch.bind('qacc_warmstart')
        self.outputs = {n: self.batch.bind(n) for n in
                        ('xpos', 'xquat', 'cvel', 'subtree_com', 'cfrc_ext', 'sensordata')}
        joint_ids = np.array([self.model.joint(n).id for n in self.dof_names])
        self.pos_addrs = torch.tensor(self.model.jnt_qposadr[joint_ids], device=self.device, dtype=torch.long)
        self.vel_addrs = torch.tensor(self.model.jnt_dofadr[joint_ids], device=self.device, dtype=torch.long)
        self.root_addr = self.model.joint('floating_base_joint').qposadr[0]
        self.root_vel_addr = self.model.joint('floating_base_joint').dofadr[0]
        self.ball_addr = self.model.joint('ball_free').qposadr[0]
        self.ball_vel_addr = self.model.joint('ball_free').dofadr[0]
        self.base_indice = self.model.body(cfg['asset']['base_name']).id
        self.head_indice = self.model.body(cfg['asset']['head_name']).id
        self.ball_indice = self.model.body('ball').id
        self.feet_indices = torch.tensor([self.model.body(n).id for n in cfg['asset']['foot_names']], device=self.device)
        self.num_bodies = self.model.nbody
        self.body_root_ids = torch.tensor(self.model.body_rootid, device=self.device, dtype=torch.long)
        self.dof_pos_limits = self._tensor(self.model.jnt_range[joint_ids])
        self.dof_vel_limits = self._tensor(self._urdf_velocity_limits())
        self.torque_limits = self._tensor(self.model.actuator_ctrlrange[:, 1])
        self.penalized_contact_indices = torch.tensor([
            b for b in range(1, self.num_bodies) if any(
                token in self.model.body(b).name for token in cfg['rewards']['penalize_contacts_on'])
        ], dtype=torch.long, device=self.device)
        self.termination_contact_indices = torch.tensor([
            b for b in range(1, self.num_bodies) if any(
                token in self.model.body(b).name for token in cfg['rewards']['terminate_contacts_on'])
        ], dtype=torch.long, device=self.device)
        gains = cfg['control']
        kp = [next(v for k, v in gains['stiffness'].items() if k in n) for n in self.dof_names]
        kd = [next(v for k, v in gains['damping'].items() if k in n) for n in self.dof_names]
        self.nominal_kp = self._tensor(kp)
        self.nominal_kd = self._tensor(kd)
        self.pd_kp = self.nominal_kp.repeat(self.num_envs, 1)
        self.pd_kd = self.nominal_kd.repeat(self.num_envs, 1)
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, device=self.device)
        self.batch.forward()
        self._init_buffers()
        self._survival_checked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_ball_idle_s = torch.zeros(self.num_envs, device=self.device)
        self.near_ball_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.far_ball_idle_s = torch.zeros_like(self.near_ball_idle_s)
        self.near_ball_stall = torch.zeros_like(self.near_ball_idle_s)
        self.far_ball_stall = torch.zeros_like(self.near_ball_idle_s)
        self.approach_window_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.approach_anchor_base = torch.zeros(self.num_envs, 2, device=self.device)
        self.approach_anchor_ball = torch.zeros_like(self.approach_anchor_base)
        self.amp_reward_multiplier = torch.ones_like(self.near_ball_idle_s)
        self.near_start_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        neutral_fk = ResetKinematics(cfg['asset']['file'], self.dof_names, cfg['asset']['foot_names'], self.device)
        neutral = neutral_fk(self.default_dof_pos)
        self.neutral_feet_pos = torch.stack([neutral[n][0] for n in cfg['asset']['foot_names']], dim=1)
        self.neutral_feet_quat = torch.stack([neutral[n][1] for n in cfg['asset']['foot_names']], dim=1)
        self._survival_pass_mass = 0.
        self._survival_trial_mass = 0.
        self.survival_rate_5s = 0.
        self._prepare_reward_function()
        self.motion_sampler = Motion(cfg, self.device)
        self.mass = self.batch.expand('body_mass')
        self.inertia = self.batch.expand('body_inertia')
        self.ipos = self.batch.expand('body_ipos')
        self.friction = self.batch.expand('geom_friction')
        self._randomize_models(torch.arange(self.num_envs, device=self.device))

    def _tensor(self, a):
        return torch.as_tensor(a, device=self.device).float()

    def _urdf_velocity_limits(self):
        import xml.etree.ElementTree as ET
        joints = {j.get('name'): j for j in ET.parse(self.cfg['asset']['file']).getroot().findall('joint')}
        return [float(joints[n].find('limit').get('velocity')) for n in self.dof_names]

    def _sensor(self, name):
        sensor = self.model.sensor(name)
        return self.outputs['sensordata'][:, sensor.adr[0]:sensor.adr[0] + sensor.dim[0]]

    def _bind_sim_buffers(self):
        self.root_states = torch.zeros(self.num_envs, 2, 13, device=self.device)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.body_states = torch.zeros(self.num_envs, self.num_bodies, 13, device=self.device)
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._sync_states()

    def _sync_states(self):
        self.dof_pos[:] = self._tensor(self.qpos[:, self.pos_addrs])
        self.dof_vel[:] = self._tensor(self.qvel[:, self.vel_addrs])
        for actor, a, v in ((0, self.root_addr, self.root_vel_addr), (1, self.ball_addr, self.ball_vel_addr)):
            self.root_states[:, actor, :3] = self._tensor(self.qpos[:, a:a + 3])
            quat = self._tensor(self.qpos[:, [a + 4, a + 5, a + 6, a + 3]])
            self.root_states[:, actor, 3:7] = quat
            self.root_states[:, actor, 7:10] = self._tensor(self.qvel[:, v:v + 3])
            self.root_states[:, actor, 10:13] = quat_rotate(quat, self._tensor(self.qvel[:, v + 3:v + 6]))
        self.body_states[:, :, :3] = self._tensor(self.outputs['xpos'])
        self.body_states[:, :, 3:7] = self._tensor(self.outputs['xquat'][:, :, [1, 2, 3, 0]])
        angular = self.outputs['cvel'][:, :, :3]
        offset = self.outputs['xpos'] - self.outputs['subtree_com'][:, self.body_root_ids]
        linear = self.outputs['cvel'][:, :, 3:] + torch.linalg.cross(angular, offset, dim=-1)
        self.body_states[:, :, 7:10] = self._tensor(linear)
        self.body_states[:, :, 10:13] = self._tensor(angular)
        # cfrc_ext includes applied pushes; the collision reward should exclude them.
        self.contact_forces[:] = self._tensor(self.outputs['cfrc_ext'][:, :, 3:] - self.xfrc[:, :, :3])

    def _randomize_models(self, ids):
        n = len(ids)
        rng = self.cfg['randomization']
        def uniform(key, shape):
            lo, hi = rng[key]['range']
            return torch.rand(shape, device=self.device) * (hi - lo) + lo
        scales = uniform('other_mass', (n, self.num_bodies))
        scales[:, self.base_indice] = uniform('base_mass', (n,))
        scales[:, 0] = 1.
        self.mass[ids] = self._tensor(self.model.body_mass) * scales
        self.inertia[ids] = self._tensor(self.model.body_inertia) * scales[:, :, None]
        self.mass[ids, self.ball_indice] = uniform('ball_mass', (n,))
        self.inertia[ids, self.ball_indice] = self._tensor(self.model.body_inertia)[self.ball_indice] * (
            self.mass[ids, self.ball_indice] / self._tensor(self.model.body_mass)[self.ball_indice])[:, None]
        com = uniform('other_com', (n, self.num_bodies, 3))
        com[:, self.base_indice] = uniform('base_com', (n, 3))
        com[:, [0, self.ball_indice]] = 0
        self.ipos[ids] = self._tensor(self.model.body_ipos) + com
        self.friction[ids, :, 0] = uniform('friction', (n, 1))
        ball_geom = self.model.geom('soccer_ball').id
        self.friction[ids, ball_geom, 0] = uniform('ball_friction', (n,))
        self.batch.set_const(ids)
        self.base_mass_scaled[torch.as_tensor(ids, device=self.device, dtype=torch.long), :3] = self._tensor(com[:, self.base_indice])
        self.base_mass_scaled[torch.as_tensor(ids, device=self.device, dtype=torch.long), 3] = self._tensor(scales[:, self.base_indice])
        self.pd_kp[ids] = self.nominal_kp * uniform('dof_stiffness', (n, self.num_dofs))
        self.pd_kd[ids] = self.nominal_kd * uniform('dof_damping', (n, self.num_dofs))

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        ids = env_ids.sort().values
        env_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._randomize_models(ids)
        self.qpos[ids] = self._tensor(self.model.qpos0)
        self.qvel[ids] = 0
        self.ctrl[ids] = 0
        self.xfrc[ids] = 0
        self.warmstart[ids] = 0
        a, v = self.root_addr, self.root_vel_addr
        yaw = torch.rand(len(ids), device=self.device) * (2 * torch.pi)
        quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.qpos[(ids[:, None], self.pos_addrs)] = self.default_dof_pos[env_ids]
        self.qpos[ids, a + 2] = self.cfg['init_state']['pos'][2]
        if self.cfg['motion']['reference_state_init']:
            mask = torch.rand(len(ids), device=self.device) < self.reference_init_probability
            reference_ids = ids[mask]
            if len(reference_ids):
                frames = self.motion_sampler.sample_batch(len(reference_ids))
                quat[mask] = frames['base_quat']
                self.qpos[(reference_ids[:, None], self.pos_addrs)] = frames['dof_pos']
                self.qvel[(reference_ids[:, None], self.vel_addrs)] = frames['dof_vel']
                self.qpos[reference_ids, a + 2] = frames['base_height'] + .01
                self.qvel[reference_ids, v:v + 3] = quat_rotate(frames['base_quat'], frames['base_lin_vel'])
                self.qvel[reference_ids, v + 3:v + 6] = frames['base_ang_vel']
        self.qpos[ids, a:a + 2] = torch.rand((len(ids), 2), device=self.device) * self._tensor([13., 8.]) - self._tensor([6.5, 4.])
        self.qpos[(ids[:, None], [a + 3, a + 4, a + 5, a + 6])] = quat[:, [3, 0, 1, 2]]
        self._reset_ball_states(ids)
        self._reset_near_start(ids)
        self.batch.forward(ids)
        self._sync_states()
        self._refresh_task_state()
        self.actions[env_ids] = 0
        self.last_actions[env_ids] = 0
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.episode_length_buf[env_ids] = 0
        self._survival_checked[env_ids] = False
        self.near_ball_idle_s[env_ids] = 0
        self.near_ball_active[env_ids] = False
        self.far_ball_idle_s[env_ids] = 0
        self.near_ball_stall[env_ids] = 0
        self.far_ball_stall[env_ids] = 0
        self.approach_window_steps[env_ids] = 0
        self.amp_reward_multiplier[env_ids] = 1
        self.buffer_count[env_ids] = 0
        self.buffer_idx[env_ids] = 0
        self.goal_cnt[env_ids] = 0
        self.ball_obs_buffer[env_ids] = 0
        self.ball_delay_steps[env_ids] = apply_randomization(
            torch.zeros(len(ids), device=self.device), self.cfg['randomization']['ball_delay_step']).round().long().clamp(0, 19)
        self.delay_steps[env_ids] = torch.randint(self.cfg['control']['decimation'], (len(ids),), device=self.device)
        self.last_base_pos[env_ids] = self.base_pos[env_ids, :2]
        self.last_ball_pos[env_ids] = self.ball_pos[env_ids, :2]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]

    def _reset_ball_states(self, ids):
        a, v = self.ball_addr, self.ball_vel_addr
        # Sample outside the robot's full leg reach plus the ball radius.
        # Independent uniform samples can otherwise start a foot inside the ball.
        robot_xy = self.qpos[ids, self.root_addr:self.root_addr + 2]
        def sample_positions(sample_ids):
            count = len(sample_ids)
            radius = self.cfg['init_state']['ball_spawn_radius'] * torch.rand(count, device=self.device).sqrt()
            angle = 2 * torch.pi * torch.rand(count, device=self.device)
            positions = radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
            field = self.spawn_group[sample_ids] == 0
            margin = self.cfg['rewards']['ball_radius']
            half_size = positions.new_tensor([7. - margin, 4.5 - margin])
            positions[field] = (torch.rand((int(field.sum()), 2), device=self.device) * 2 - 1) * half_size
            return positions
        ball_xy = sample_positions(ids)
        for _ in range(32):
            overlap = torch.linalg.vector_norm(ball_xy - robot_xy, dim=1) < .65
            if not overlap.any():
                break
            ball_xy[overlap] = sample_positions(ids[overlap])
        else:
            raise RuntimeError('Could not sample a collision-free ball spawn')
        self.qpos[ids, a:a + 2] = ball_xy
        self.qpos[ids, a + 2:a + 7] = self._tensor([0.11, 1, 0, 0, 0])
        self.qvel[ids, v:v + 6] = 0

    @property
    def near_start_fraction(self):
        cfg = self.cfg.get('curriculum', {}).get('near_start', {})
        if self.cfg['init_state'].get('spawn_mixture') is not None:
            return self.cfg['init_state']['spawn_mixture'][2]
        if not cfg.get('enabled', False) or self.balance_phase:
            return 0.
        return cfg['initial_fraction'] + self.near_start_progress * (cfg['final_fraction'] - cfg['initial_fraction'])

    def near_start_state(self):
        return dict(version=self.cfg['curriculum'].get('near_start', {}).get('version'),
                    progress=float(self.near_start_progress), success_rate=float(self.near_start_success_rate),
                    trials=int(self.near_start_trials), success_mass=float(self._near_start_success_mass),
                    trial_mass=float(self._near_start_trial_mass), last_adjustment=self._near_start_last_adjustment,
                    last_trial_count=self._near_start_last_trial_count,
                    full_success_rate=float(self.full_start_success_rate), full_success_mass=float(self._full_start_success_mass),
                    full_trial_mass=float(self._full_start_trial_mass))

    def restore_near_start_state(self, state):
        self.near_start_progress = state.get('progress', 0.)
        self.near_start_success_rate = state.get('success_rate', 0.)
        self.near_start_trials = state.get('trials', 0)
        self._near_start_success_mass = state.get('success_mass', 0.)
        self._near_start_trial_mass = state.get('trial_mass', 0.)
        self._near_start_last_adjustment = state.get('last_adjustment')
        self._near_start_last_trial_count = state.get('last_trial_count', 0)
        self.full_start_success_rate = state.get('full_success_rate', 0.)
        self._full_start_success_mass = state.get('full_success_mass', 0.)
        self._full_start_trial_mass = state.get('full_trial_mass', 0.)

    def _reset_near_start(self, ids):
        """Neutral, collision-free last-step practice, mixed with full-field resets."""
        if self.cfg['init_state'].get('spawn_mixture') is not None:
            self.near_start_episode[ids] = self.spawn_group[ids] == 2
        else:
            self.near_start_episode[ids] = torch.rand(len(ids), device=self.device) < self.near_start_fraction
        selected = ids[self.near_start_episode[ids]]
        if not len(selected):
            return
        cfg = self.cfg['curriculum']['near_start']
        count = len(selected)
        ball = self.qpos[selected, self.ball_addr:self.ball_addr + 3]
        target = ball.new_tensor([7., 0.]) - ball[:, :2]
        spread = cfg['yaw_error_degrees'] + self.near_start_progress * (cfg['final_yaw_error_degrees'] - cfg['yaw_error_degrees'])
        yaw = torch.atan2(target[:, 1], target[:, 0]) + (torch.rand(count, device=self.device) * 2 - 1) * np.deg2rad(spread)
        quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        foot = torch.randint(2, (count,), device=self.device)
        foot_offset = quat_rotate(quat, self.neutral_feet_pos[selected, foot])
        foot_quat = quat_mul(quat, self.neutral_feet_quat[selected, foot])
        low_gap, high_gap = cfg['gap_range']
        high_gap += self.near_start_progress * (cfg['final_gap_max'] - high_gap)
        gap = low_gap + torch.rand(count, device=self.device) * (high_gap - low_gap)
        height = self.cfg['init_state']['pos'][2]
        top = self.cfg['asset']['foot_collision_bounds'][1]
        vertical = (ball[:, 2] - height - foot_offset[:, 2] - top[2]).clamp_min(0)
        horizontal = ((gap + self.cfg['rewards']['ball_radius']).square() - vertical.square()).clamp_min(0).sqrt()
        relative_ball = torch.zeros(count, 3, device=self.device)
        relative_ball[:, 0] = top[0] + horizontal
        world_offset = quat_rotate(foot_quat, relative_ball)
        a = self.root_addr
        self.qpos[selected, a:a + 2] = ball[:, :2] - foot_offset[:, :2] - world_offset[:, :2]
        self.qpos[selected, a + 2] = height
        self.qpos[(selected[:, None], [a + 3, a + 4, a + 5, a + 6])] = quat[:, [3, 0, 1, 2]]
        self.qpos[(selected[:, None], self.pos_addrs)] = self.default_dof_pos[selected]
        self.qvel[selected] = 0

    def _record_near_start_trials(self):
        resolved = self.near_start_episode & self.reset_buf
        count = int(resolved.sum())
        if count:
            successes = self.extras['success'][resolved].sum().item()
            decay = float(np.exp(-count / 512.))
            self._near_start_success_mass = decay * self._near_start_success_mass + successes
            self._near_start_trial_mass = decay * self._near_start_trial_mass + count
            self.near_start_success_rate = self._near_start_success_mass / self._near_start_trial_mass
            self.near_start_trials += count
        resolved = ~self.near_start_episode & self.reset_buf
        count = int(resolved.sum())
        if count:
            decay = float(np.exp(-count / 512.))
            self._full_start_success_mass = decay * self._full_start_success_mass + self.extras['success'][resolved].sum().item()
            self._full_start_trial_mass = decay * self._full_start_trial_mass + count
            self.full_start_success_rate = self._full_start_success_mass / self._full_start_trial_mass

    def _compute_amp_obs(self):
        return torch.cat((self.projected_gravity, self.base_lin_vel, self.base_ang_vel,
                          self.dof_pos, self.dof_vel * self.cfg['normalization']['dof_vel'],
                          quat_rotate_inverse(self.base_quat, self.feet_pos[:, 0] - self.base_pos),
                          quat_rotate_inverse(self.base_quat, self.feet_pos[:, 1] - self.base_pos)), -1)

    def _check_termination(self):
        super()._check_termination()
        self.extras['success'] *= 1. - self.extras['fall']
        self.time_out_buf[:] = ((self.episode_length_buf > np.ceil(self.cfg['rewards']['episode_length_s'] / self.dt)) |
                                self.extras['success'].bool()) & ~self.extras['fall'].bool()
        self.extras['termination_height'] = (self.base_pos[:, 2] < self.cfg['rewards']['terminate_height']).float()
        self.extras['termination_velocity'] = (
            self.root_states[:, 0, 7:13].square().sum(-1) > self.cfg['rewards']['terminate_vel']).float()

    def _reward_action_rate(self):
        return (self.last_actions - self.actions).square().sum(-1)

    def set_training_iteration(self, iteration):
        curriculum = self.cfg.get('curriculum', {})
        near_cfg = curriculum.get('near_start', {})
        if (near_cfg.get('enabled', False) and near_cfg.get('adaptive', True) and self.near_start_trials - self._near_start_last_trial_count >= near_cfg['min_completed_trials'] and
            (self._near_start_last_adjustment is None or iteration - self._near_start_last_adjustment >= near_cfg['adjust_interval_iterations'])):
            if self.near_start_success_rate >= near_cfg['advance_success_rate'] and self.survival_rate_5s >= .8:
                self.near_start_progress = min(1., self.near_start_progress + near_cfg['progress_step'])
            elif self.near_start_success_rate < near_cfg['retreat_success_rate']:
                self.near_start_progress = max(0., self.near_start_progress - near_cfg['progress_step'])
            self._near_start_last_adjustment = iteration
            self._near_start_last_trial_count = self.near_start_trials
        until = curriculum.get('balance_until_iteration', 0)
        minimum_fraction = curriculum.get('release_active_5s_fraction', 0.)
        active_fraction = (self.episode_length_buf * self.dt >= 5).float().mean().item()
        quality = min(active_fraction, self.survival_rate_5s) if curriculum.get('use_cohort_gate', False) else active_fraction
        if self.balance_release_iteration is None and iteration >= until and quality >= minimum_fraction:
            self.balance_release_iteration = iteration
        balance = self.balance_release_iteration is None
        changed = balance != self.balance_phase
        self.balance_phase = balance
        self.cfg['motion']['reference_state_init'] = not balance
        transition = max(1, curriculum.get('soccer_transition_iterations', 1))
        if balance:
            progress = 0.
        elif curriculum.get('adaptive', False):
            progress = self.soccer_progress
            if iteration != self._curriculum_iteration:
                interval = max(1, curriculum.get('adjust_interval_iterations', 1))
                self._curriculum_quality_window.append(quality)
                self._curriculum_quality_window = self._curriculum_quality_window[-interval:]
                if (len(self._curriculum_quality_window) >= interval and
                    (self._curriculum_last_adjustment is None or iteration - self._curriculum_last_adjustment >= interval)):
                    mean_quality = sum(self._curriculum_quality_window) / len(self._curriculum_quality_window)
                    increment = min(interval / transition, curriculum.get('max_progress_step', 1.))
                    reliable_scoring = self.recent_success_rate >= .1
                    if mean_quality >= curriculum.get('advance_active_5s_fraction', .7) or reliable_scoring:
                        progress = min(1., progress + increment)
                    elif mean_quality < curriculum.get('retreat_active_5s_fraction', .4):
                        progress = max(0., progress - 2. * increment)
                    self._curriculum_last_adjustment = iteration
        else:
            progress = min(1., (iteration - self.balance_release_iteration + 1) / transition)
        self._curriculum_iteration = iteration
        self.soccer_progress = progress
        self.amp_reward_scale = progress
        self.reference_init_probability = progress
        for name in ('goal', 'ball_distance', 'goal_distance', 'kick_ball', 'side_kick_ball',
                     'face_ball_pitch', 'face_ball_yaw', 'pos_still', 'near_ball_stall'):
            if name in self.reward_scales:
                self.reward_scales[name] = self.cfg['rewards']['scales'][name] * self.dt * progress
        return changed

    def _reward_upright(self):
        return self.projected_gravity[:, :2].square().sum(-1)

    def _record_survival_trials(self):
        reached = self.episode_length_buf * self.dt >= 5
        resolved = (reached | self.reset_buf) & ~self._survival_checked
        n = int(resolved.sum())
        if n:
            passed = int((resolved & ((reached & ~self.extras['fall'].bool()) | self.extras['success'].bool())).sum())
            decay = np.exp(-n / 4096.)
            self._survival_pass_mass = decay * self._survival_pass_mass + passed
            self._survival_trial_mass = decay * self._survival_trial_mass + n
            self.survival_rate_5s = self._survival_pass_mass / self._survival_trial_mass
            self._survival_checked[resolved] = True

    def _reward_base_height(self):
        return (self.base_pos[:, 2] - self.cfg['init_state']['pos'][2]).square()

    def _reward_stance_pose(self):
        return (self.dof_pos - self.default_dof_pos).square().mean(-1) * (1. - self.amp_reward_scale)

    def _foot_ball_gap(self):
        """Sphere-to-foot collision AABB gap in each foot's local frame.

        The box is conservative: proximity is not an exact contact detector.
        Bounds come from the official Q1 foot collision mesh, not leg length.
        """
        relative = self.ball_pos[:, None, :] - self.feet_pos
        local = quat_rotate_inverse(self.feet_quat.flatten(0, 1), relative.flatten(0, 1)).view(self.num_envs, 2, 3)
        low = self._tensor(self.cfg['asset']['foot_collision_bounds'][0])
        high = self._tensor(self.cfg['asset']['foot_collision_bounds'][1])
        closest = local.clamp(min=low, max=high)
        return (torch.linalg.vector_norm(local - closest, dim=-1) - self.cfg['rewards']['ball_radius']).clamp_min(0).amin(-1)

    def _update_soccer_feedback(self):
        cfg = self.cfg['rewards']
        gap = self._foot_ball_gap()
        # Separate entry/exit distances prevent threshold jitter from repeatedly
        # switching reward modes and granting a fresh preparation period.
        near = gap < (cfg['near_ball_gap'] + self.near_ball_active.float() * cfg['near_ball_hysteresis'])
        self.near_ball_active[:] = near
        goal = self.ball_pos.new_tensor([7., 0.])
        direction = goal - self.last_ball_pos
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        displacement = ((self.ball_pos[:, :2] - self.last_ball_pos) * direction).sum(-1)
        self.ball_goal_speed = displacement / self.dt
        ball_progress = self.ball_goal_speed > cfg['ball_progress_speed']
        # Once near the ball, foot/body wiggling cannot restart the grace timer.
        self.near_ball_idle_s[:] = torch.where(near & ~ball_progress, self.near_ball_idle_s + self.dt, 0.)
        grace = cfg['near_ball_grace_s']
        self.near_ball_stall[:] = ((self.near_ball_idle_s - grace) / cfg['near_ball_stall_ramp_s']).clamp(0, 1)
        # Compare net progress over a complete window. Per-step velocity lets
        # submillimetre rocking repeatedly clear the idle timer.
        start = self.approach_window_steps == 0
        self.approach_anchor_base[start] = self.base_pos[start, :2]
        self.approach_anchor_ball[start] = self.ball_pos[start, :2]
        self.approach_window_steps += 1
        window_steps = max(1, round(cfg['approach_window_s'] / self.dt))
        finished = self.approach_window_steps >= window_steps
        net_approach = (torch.linalg.vector_norm(self.approach_anchor_base - self.approach_anchor_ball, dim=-1)
                        - torch.linalg.vector_norm(self.base_pos[:, :2] - self.approach_anchor_ball, dim=-1))
        productive = net_approach >= cfg['approach_progress_speed'] * cfg['approach_window_s']
        self.far_ball_idle_s[:] = torch.where(
            near | (finished & productive), 0.,
            self.far_ball_idle_s + (finished & ~near).float() * window_steps * self.dt)
        self.approach_window_steps[finished] = 0
        pending_s = self.approach_window_steps * self.dt
        self.far_ball_stall[:] = torch.where(
            near, 0., ((self.far_ball_idle_s + pending_s - cfg['far_ball_grace_s']) /
                       cfg['far_ball_stall_ramp_s']).clamp(0, 1))
        self.amp_reward_multiplier[:] = 1. - torch.maximum(self.near_ball_stall, self.far_ball_stall)
        self.foot_ball_gap = gap
        self.extras['soccer_episode'] = {
            'near_ball_s': near.float() * self.dt,
            'near_ball_stall_s': (self.near_ball_stall > 0).float() * self.dt,
            'far_ball_stall_s': (self.far_ball_stall > 0).float() * self.dt,
            'ball_forward_distance': displacement.clamp_min(0),
        }

    def _compute_reward(self):
        self._update_soccer_feedback()
        super()._compute_reward()
        # Resets happen before Runner consumes extras. Keep the terminal frame's
        # multiplier independent of the live per-world buffers reset below.
        self.extras['amp_reward_multiplier'] = self.amp_reward_multiplier.clone()

    def _reward_pos_still(self):
        return self.far_ball_stall

    def _reward_near_ball_stall(self):
        return self.near_ball_stall

    def _kick_reward_components(self):
        # Reward goal-directed ball movement only within plausible foot reach.
        # A stationary ball and a fast foot alone earn nothing.
        near_contact = self.foot_ball_gap < self.cfg['rewards']['kick_ball_gap']
        upright = self.projected_gravity[:, 2] < -.8
        valid = near_contact & upright & ~self.extras['fall'].bool()
        foot = torch.linalg.vector_norm(self.feet_pos - self.ball_pos[:, None, :], dim=-1).argmin(-1)
        rows = torch.arange(self.num_envs, device=self.device)
        velocity = self.feet_vel[rows, foot]
        yaw = self.feet_yaw[rows, foot]
        forward = torch.cos(yaw) * velocity[:, 0] + torch.sin(yaw) * velocity[:, 1]
        lateral = -torch.sin(yaw) * velocity[:, 0] + torch.cos(yaw) * velocity[:, 1]
        magnitude = forward.abs() + lateral.abs()
        motion = valid.float() * (magnitude > .05).float() * self.ball_goal_speed.clamp(0, self.cfg['rewards']['max_kick_vel'])
        # Split diagonal kicks instead of giving both full rewards. Both terms
        # still require actual goal-directed ball movement, never an air kick.
        return (motion * forward.clamp_min(0) / magnitude.clamp_min(1e-6),
                motion * lateral.abs() / magnitude.clamp_min(1e-6))

    def _reward_kick_ball(self):
        return self._kick_reward_components()[0]

    def _reward_side_kick_ball(self):
        return self._kick_reward_components()[1]

    def step(self, actions):
        self.actions[:] = actions.detach().clamp(-self.cfg['normalization']['clip_actions'], self.cfg['normalization']['clip_actions'])
        targets = (self.default_dof_pos + self.action_scale * self.actions)
        delayed = self.last_dof_targets.clone()
        delays = self.delay_steps
        limits = self.torque_limits
        self._perturb()
        for i in range(self.cfg['control']['decimation']):
            delayed.copy_(torch.where((delays == i)[:, None], targets, delayed))
            torque = self.pd_kp * (delayed - self.qpos[:, self.pos_addrs]) - self.pd_kd * self.qvel[:, self.vel_addrs]
            self.ctrl[:] = torch.clamp(torque, -limits, limits)
            self.batch.step()
        self.last_dof_targets[:] = self._tensor(delayed)
        # MuJoCo step outputs lag integration by one substep. Refresh once at the
        # control boundary, keeping every observation at the same simulation time.
        self.batch.forward()
        if not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all():
            raise FloatingPointError('Non-finite batched simulation state')
        self._sync_states()
        self._refresh_task_state()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._check_termination()
        self._record_near_start_trials()
        self._record_survival_trials()
        if self.reset_buf.any():
            rate = self.extras['success'][self.reset_buf].mean().item()
            self.recent_success_rate = .95 * self.recent_success_rate + .05 * rate
        self._compute_reward()
        self.extras['last_amp_obs'] = self._compute_amp_obs().clone()
        self.extras['time_outs'] = self.time_out_buf.clone()
        self._reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        self._compute_observations()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_yaw[:] = self.feet_yaw
        self.last_base_pos[:] = self.base_pos[:, :2]
        self.last_ball_pos[:] = self.ball_pos[:, :2]
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _perturb(self):
        difficulty = self.amp_reward_scale
        if difficulty <= 0:
            self.xfrc[:] = 0
            self.pushing_forces[:] = 0
            return
        cfg = self.cfg['randomization']
        step = self.common_step_counter
        if step and step % max(1, round(cfg['kick_interval_s'] / self.dt)) == 0:
            v = self.root_vel_addr
            self.qvel[:, v:v + 3] += difficulty * torch.randn((self.num_envs, 3), device=self.device) * cfg['kick_lin_vel']['range'][1]
            self.qvel[:, v + 3:v + 6] += difficulty * torch.randn((self.num_envs, 3), device=self.device) * cfg['kick_ang_vel']['range'][1]
        interval = max(1, round(cfg['push_interval_s'] / self.dt))
        if step % interval == 0:
            self.xfrc[:, self.base_indice, :3] = difficulty * torch.randn((self.num_envs, 3), device=self.device) * cfg['push_force']['range'][1]
            self.xfrc[:, self.base_indice, 3:] = difficulty * torch.randn((self.num_envs, 3), device=self.device) * cfg['push_torque']['range'][1]
        elif step % interval == round(cfg['push_duration_s'] / self.dt):
            self.xfrc[:] = 0
        self.pushing_forces[:] = self._tensor(self.xfrc[:, :, :3])
        mask = torch.rand(self.num_envs, device=self.device) < cfg['kick_ball_prob']
        self.qvel[mask, self.ball_vel_addr:self.ball_vel_addr + 2] += torch.randn((int(mask.sum()), 2), device=self.device) * cfg['kick_ball_vel']
        ids = (torch.rand(self.num_envs, device=self.device) < cfg['reset_ball_prob']).nonzero().flatten()
        if len(ids):
            self._reset_ball_states(ids)
            idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            self.ball_obs_buffer[idx] = 0
            self.last_ball_pos[idx] = self._tensor(self.qpos[ids, self.ball_addr:self.ball_addr + 2])

    def render(self):
        pass  # Use play_mujoco.py for interactive policy evaluation.
