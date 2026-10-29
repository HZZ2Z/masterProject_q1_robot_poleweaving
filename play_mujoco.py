# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run a Q1/G1/T1 soccer policy in MuJoCo using the task configuration.

Loads the robot scene, reference resets, and observation dimensions from the
selected task configuration. Run from the repository root.
"""

import argparse
import os
import time
from pathlib import Path

# The packaged GLFW selects a Wayland-only library on Wayland desktops.
# Use its X11 variant when DISPLAY is available so window decorations work.
if os.environ.get('DISPLAY'):
    os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from utils.model import ActorCritic


def quat_rotate(q, v):
    q_vec = q[:3]
    q_w = q[3]
    return v * (2.0 * q_w**2 - 1.0) + 2.0 * q_w * np.cross(q_vec, v) + 2.0 * q_vec * np.dot(q_vec, v)


def quat_rotate_inverse(q, v):
    q_vec = q[:3]
    q_w = q[3]
    return v * (2.0 * q_w**2 - 1.0) - 2.0 * q_w * np.cross(q_vec, v) + 2.0 * q_vec * np.dot(q_vec, v)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def camera_frustum_vertices(model, data, cfg, distance=2.5):
    """World-space optical frustum, using exactly the observation convention."""
    origin = data.sensor('camera_position').data.copy()
    camera = data.sensor('camera_orientation').data[[1, 2, 3, 0]]
    optical = quat_mul(camera, [.5] * 4)
    half_width, half_height = distance * np.tan(np.deg2rad(cfg['asset']['camera_fov']) / 2)
    corners = np.array([origin + quat_rotate(optical, np.array([x, y, distance]))
                        for x, y in ((-half_width, -half_height), (half_width, -half_height),
                                     (half_width, half_height), (-half_width, half_height))])
    return origin, corners


def draw_camera_frustum(scene, model, data, cfg, distance=2.5):
    """Viewer-only transparent faces; never contributes simulation geometry."""
    origin, corners = camera_frustum_vertices(model, data, cfg, distance)
    if scene.ngeom + 15 > scene.maxgeom:
        return

    def initialize(kind, size, pos, mat, rgba):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                           np.asarray(mat, dtype=float).reshape(9), np.asarray(rgba, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = .45
        scene.ngeom += 1
        return geom

    def triangle(a, b, c, alpha):
        u, v = b - a, c - a
        normal = np.cross(u, v)
        mat = np.column_stack((u / np.linalg.norm(u), v / np.linalg.norm(v), normal / np.linalg.norm(normal)))
        initialize(mujoco.mjtGeom.mjGEOM_TRIANGLE, [np.linalg.norm(u), np.linalg.norm(v), 0], a, mat,
                   [.1, .7, 1., alpha])

    for i in range(4):
        triangle(origin, corners[i], corners[(i + 1) % 4], .075)
    triangle(corners[0], corners[1], corners[2], .035)
    triangle(corners[0], corners[2], corners[3], .035)
    for a, b in [(origin, corner) for corner in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]:
        geom = initialize(mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], [0, 0, 0], np.eye(3), [.15, .75, 1., .8])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., a, b)
    initialize(mujoco.mjtGeom.mjGEOM_SPHERE, [.012] * 3, origin, np.eye(3), [1., .65, .1, 1.])


def latest_g1_checkpoint():
    paths = list(Path('logs').glob('g1_13dof_*/logs/*/nn/model_*.pth'))
    if not paths:
        raise FileNotFoundError('No G1 13-DoF checkpoint found; pass --checkpoint')
    return max(paths, key=lambda path: path.stat().st_mtime)


class SoccerPlayer:
    def __init__(self, checkpoint=None, seed=42, reset_mode='reference', task='Q1', training_spawns=True):
        with open(f'envs/{task}.yaml', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.task = task
        self.training_spawns = training_spawns and task == 'Q1' and self.cfg['init_state'].get('spawn_mixture') is not None
        self.spawn_kind = 1
        self.num_actions = self.cfg['env']['num_actions']
        self.rng = np.random.default_rng(seed)
        self.reset_mode = reset_mode
        self.reference_frames = None
        if reset_mode == 'reference' and task != 'Q1':
            files = sorted(Path(self.cfg['motion']['motion_path']).glob('*.csv'))
            if not files:
                raise FileNotFoundError(f'{task} reference motion CSV files are missing')
            self.reference_frames = np.concatenate([
                np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)[:-1]
                for path in files
            ])
        if task == 'Q1' and reset_mode == 'reference':
            from utils.motion import Motion
            torch.manual_seed(seed)
            self.q1_motion = Motion(self.cfg, 'cpu')
        if checkpoint and checkpoint != '-1':
            path = Path(checkpoint)
        elif task == 'G1':
            path = latest_g1_checkpoint()
        else:
            paths = list(Path(self.cfg['runner']['save_path']).glob('**/nn/model_*.pth'))
            if not paths:
                raise FileNotFoundError(f'No {task} checkpoint found; train first or pass --checkpoint')
            path = max(paths, key=lambda p: p.stat().st_mtime)
        self.policy = ActorCritic(self.cfg)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        self.policy.load_state_dict(saved['model'], strict=True)
        self.policy.eval()
        print(f'Loading {task} policy from {path}', flush=True)

        self.model = mujoco.MjModel.from_xml_path(self.cfg['asset']['mujoco_file'])
        self.model.opt.timestep = self.cfg['sim']['dt']
        self.data = mujoco.MjData(self.model)
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        if 'dof_names' in self.cfg['asset'] and names != self.cfg['asset']['dof_names']:
            raise ValueError(f'MuJoCo actuator order differs from {task} training: {names}')
        if self.model.nu != self.cfg['env']['num_actions']:
            raise ValueError(f'MuJoCo actuator count differs from the {task} policy')
        joints = self.model.actuator_trnid[:, 0]
        self.pos_addrs = self.model.jnt_qposadr[joints]
        self.vel_addrs = self.model.jnt_dofadr[joints]
        free_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_ids) != 2:
            raise ValueError('Soccer scene must have a free robot base and ball')
        self.root_joint, self.ball_joint = free_ids
        self.root_addr = self.model.jnt_qposadr[self.root_joint]
        self.ball_addr = self.model.jnt_qposadr[self.ball_joint]
        self.ball_vel_addr = self.model.jnt_dofadr[self.ball_joint]
        self.effort_limits = np.abs(self.model.actuator_ctrlrange[:, 1])
        if np.any(self.effort_limits <= 0):
            raise ValueError(f'{task} model is missing joint effort limits')

        defaults = self.cfg['init_state']['default_joint_angles']
        gains = self.cfg['control']
        self.default_pos = np.array([defaults.get(name, defaults['default']) for name in names], dtype=np.float32)
        self.kp = np.array([next(v for key, v in gains['stiffness'].items() if key in name) for name in names], dtype=np.float32)
        self.kd = np.array([next(v for key, v in gains['damping'].items() if key in name) for name in names], dtype=np.float32)
        self.action_scale = np.asarray(gains['action_scale'], dtype=np.float32)
        self.decimation = gains['decimation']
        self.actions = np.zeros(self.model.nu, dtype=np.float32)
        self.targets = self.default_pos.copy()
        self.history = np.zeros((self.cfg['runner']['num_stack'], self.cfg['env']['num_observations']), dtype=np.float32)
        self.ball_history = np.zeros((20, 3), dtype=np.float32)
        self.ball_delay = 0
        self.physics_steps = 0
        self.total_steps = 0
        self.control_steps = 0
        self.goals = 0
        self.outs = 0
        self.falls = 0
        self.timeouts = 0
        self.reset_robot()

    def reset_ball(self):
        a = self.ball_addr
        if self.task == 'Q1':
            for _ in range(32):
                if self.spawn_kind == 0:
                    margin = self.cfg['rewards']['ball_radius']
                    xy = self.rng.uniform([-7 + margin, -4.5 + margin], [7 - margin, 4.5 - margin])
                else:
                    radius = self.cfg['init_state']['ball_spawn_radius'] * np.sqrt(self.rng.random())
                    angle = self.rng.uniform(0, 2 * np.pi)
                    xy = radius * np.array([np.cos(angle), np.sin(angle)])
                if np.linalg.norm(xy - self.data.qpos[self.root_addr:self.root_addr + 2]) >= .65:
                    break
            else:
                raise RuntimeError('Could not sample collision-free Q1 ball spawn')
            self.data.qpos[a:a + 7] = [*xy, .11, 1, 0, 0, 0]
        else:
            self.data.qpos[a:a + 7] = [self.rng.uniform(-7, 7), self.rng.uniform(-4.5, 4.5), 0.1, 1, 0, 0, 0]
        self.data.qvel[self.ball_vel_addr:self.ball_vel_addr + 6] = 0
        self.ball_history[:] = 0
        self.ball_delay = int(np.clip(np.rint(self.rng.normal(6, 1)), 0, len(self.ball_history) - 1))
        mujoco.mj_forward(self.model, self.data)

    def reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.training_spawns:
            self.spawn_kind = int(self.rng.choice(3, p=self.cfg['init_state']['spawn_mixture']))
        a = self.root_addr
        yaw = self.rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([0., 0., np.sin(yaw / 2), np.cos(yaw / 2)])
        base_xy = [self.rng.uniform(-7, 7), self.rng.uniform(-4.5, 4.5)]
        if self.task == 'Q1':
            base_xy = self.rng.uniform([-6.5, -4.], [6.5, 4.])
        self.data.qvel[:] = 0
        reference = self.reset_mode == 'reference' and not (self.training_spawns and self.spawn_kind == 2)
        if reference and self.task == 'Q1':
            frame = self.q1_motion.sample_batch(1)
            base_quat = frame['base_quat'][0].numpy()
            self.data.qpos[a:a + 7] = [*base_xy, frame['base_height'][0].item() + .01,
                                     *base_quat[[3, 0, 1, 2]]]
            self.data.qpos[self.pos_addrs] = frame['dof_pos'][0].numpy()
            v = self.model.jnt_dofadr[self.root_joint]
            self.data.qvel[v:v + 3] = quat_rotate(base_quat, frame['base_lin_vel'][0].numpy())
            self.data.qvel[v + 3:v + 6] = frame['base_ang_vel'][0].numpy()
            self.data.qvel[self.vel_addrs] = frame['dof_vel'][0].numpy()
        elif reference:
            frame = self.reference_frames[self.rng.integers(len(self.reference_frames))]
            base_quat = quat_mul(yaw_quat, frame[3:7])
            self.data.qpos[a:a + 7] = [*base_xy, frame[2] + 0.01,
                                         base_quat[3], base_quat[0], base_quat[1], base_quat[2]]
            start = self.cfg['motion'].get('dof_start_index', 2)
            count = self.num_actions - start
            self.data.qpos[self.pos_addrs] = self.default_pos
            self.data.qpos[self.pos_addrs[start:]] = frame[13:13 + count]
            base_vel_addr = self.model.jnt_dofadr[self.root_joint]
            self.data.qvel[base_vel_addr:base_vel_addr + 3] = quat_rotate(base_quat, frame[7:10])
            self.data.qvel[base_vel_addr + 3:base_vel_addr + 6] = frame[10:13]
            self.data.qvel[self.vel_addrs[start:]] = frame[13 + count:13 + 2 * count]
        else:
            self.data.qpos[a:a + 7] = [*base_xy, self.cfg['init_state']['pos'][2],
                                         yaw_quat[3], yaw_quat[0], yaw_quat[1], yaw_quat[2]]
            self.data.qpos[self.pos_addrs] = self.default_pos
        self.data.ctrl[:] = 0
        self.actions[:] = 0
        self.targets[:] = self.default_pos
        self.history[:] = 0
        self.physics_steps = 0
        self.control_steps = 0
        self.goal_hold_steps = 0
        self.reset_ball()
        if self.training_spawns and self.spawn_kind == 2:
            self._place_near_ball()

    def _place_near_ball(self):
        cfg = self.cfg['curriculum']['near_start']
        ball = self.data.qpos[self.ball_addr:self.ball_addr + 3]
        yaw = np.arctan2(-ball[1], 7 - ball[0]) + self.rng.uniform(-1, 1) * np.deg2rad(cfg['yaw_error_degrees'])
        self.data.qpos[self.root_addr + 3:self.root_addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        foot = self.model.body(self.cfg['asset']['foot_names'][self.rng.integers(2)]).id
        top = self.cfg['asset']['foot_collision_bounds'][1]
        gap = self.rng.uniform(*cfg['gap_range'])
        vertical = max(0, ball[2] - self.data.xpos[foot, 2] - top[2])
        horizontal = np.sqrt(max(0, (gap + self.cfg['rewards']['ball_radius'])**2 - vertical**2))
        offset = self.data.xmat[foot].reshape(3, 3) @ [top[0] + horizontal, 0, 0]
        self.data.qpos[self.root_addr:self.root_addr + 2] += ball[:2] - self.data.xpos[foot, :2] - offset[:2]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _q1_episode_end(self):
        """Same terminal conditions and goal hold duration as the Q1 task."""
        pos = self.data.qpos[self.root_addr:self.root_addr + 3]
        v = self.model.jnt_dofadr[self.root_joint]
        from envs.q1 import FlatTerrain
        lower = -.9 * FlatTerrain.border_size
        upper_x = FlatTerrain.env_width + .9 * FlatTerrain.border_size
        upper_y = FlatTerrain.env_length + .9 * FlatTerrain.border_size
        fall = (pos[2] < self.cfg['rewards']['terminate_height'] or
                np.square(self.data.qvel[v:v + 6]).sum() > self.cfg['rewards']['terminate_vel'] or
                pos[0] < lower or pos[0] > upper_x or pos[1] < lower or pos[1] > upper_y)
        if fall:
            return 'fall'
        ball = self.data.qpos[self.ball_addr:self.ball_addr + 3]
        goal = ball[0] > 7 and abs(ball[1]) < 1.3
        self.goal_hold_steps = self.goal_hold_steps + 1 if goal else 0
        if self.goal_hold_steps >= 50:
            return 'goal'
        if ball[0] < -7 or (ball[0] > 7 and abs(ball[1]) > 1.3) or abs(ball[1]) > 4.5:
            return 'out'
        if self.control_steps * self.model.opt.timestep * self.decimation > self.cfg['rewards']['episode_length_s']:
            return 'timeout'
        return None

    def observation(self):
        cfg = self.cfg
        base_pos = self.data.sensor('position').data.copy()
        base_quat_wxyz = self.data.sensor('orientation').data.copy()
        base_quat = base_quat_wxyz[[1, 2, 3, 0]]
        camera_pos = self.data.sensor('camera_position').data.copy()
        camera_quat = self.data.sensor('camera_orientation').data[[1, 2, 3, 0]].copy()
        ball_pos = self.data.qpos[self.ball_addr:self.ball_addr + 3].copy()
        sin_yaw = 2 * (base_quat[3] * base_quat[2] + base_quat[0] * base_quat[1])
        cos_yaw = base_quat[3]**2 + base_quat[0]**2 - base_quat[1]**2 - base_quat[2]**2
        yaw = np.arctan2(sin_yaw, cos_yaw)
        rotate_xy = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        relative_ball = rotate_xy @ (ball_pos[:2] - base_pos[:2])
        relative_goal = rotate_xy @ (np.array([7., 0.]) - base_pos[:2])
        ball_camera = quat_rotate_inverse(quat_mul(camera_quat, [0.5] * 4), ball_pos - camera_pos)
        h_fov, v_fov = np.deg2rad(cfg['asset']['camera_fov']) / 2
        in_cone = (ball_camera[2] > 0 and
                   abs(np.arctan2(ball_camera[0], ball_camera[2])) < h_fov and
                   abs(np.arctan2(ball_camera[1], ball_camera[2])) < v_fov)
        distance = np.linalg.norm(relative_ball)
        if in_cone:
            visible = self.rng.random() < max(0., 2.475 - 0.225 * max(distance, 7.))
        else:
            visible = relative_ball[0] > -2 and self.rng.random() < 0.01
        self.ball_history[1:] = self.ball_history[:-1].copy()
        if self.control_steps % 2 == 0:
            noise_std = 0.149 + 0.124 * distance
            self.ball_history[0, :2] = (relative_ball + self.rng.normal(0., noise_std, 2)) if visible else 0
            self.ball_history[0, 2] = float(visible)
        else:
            self.ball_history[0] = self.ball_history[1]

        obs = np.zeros(cfg['env']['num_observations'], dtype=np.float32)
        norm = cfg['normalization']
        obs[:3] = quat_rotate_inverse(base_quat, np.array([0., 0., -1.])) * norm['gravity']
        obs[3:6] = self.data.sensor('angular-velocity').data * norm['ang_vel']
        obs[6:9] = self.ball_history[self.ball_delay]
        obs[9:11] = relative_goal
        obs[11:13] = [np.cos(yaw), np.sin(yaw)]
        n = self.num_actions
        obs[13:13 + n] = (self.data.qpos[self.pos_addrs] - self.default_pos) * norm['dof_pos']
        obs[13 + n:13 + 2 * n] = self.data.qvel[self.vel_addrs] * norm['dof_vel']
        obs[13 + 2 * n:13 + 3 * n] = self.actions
        return obs

    def step(self, auto_reset=True):
        if self.task == 'Q1':
            if self.control_steps and self.physics_steps % self.decimation == 0:
                cause = self._q1_episode_end()
                if cause:
                    if not auto_reset:
                        return cause
                    attribute = {'fall': 'falls', 'goal': 'goals', 'out': 'outs', 'timeout': 'timeouts'}[cause]
                    setattr(self, attribute, getattr(self, attribute) + 1)
                    self.reset_robot()
                    print(f'Goal: {self.goals}, Out: {self.outs}, Fall: {self.falls}, Timeout: {self.timeouts}', flush=True)
        elif self.data.sensor('position').data[2] < self.cfg['rewards']['terminate_height']:
            self.falls += 1
            self.reset_robot()
            print(f'Goal: {self.goals}, Out: {self.outs}, Fall: {self.falls}', flush=True)
        else:
            ball = self.data.qpos[self.ball_addr:self.ball_addr + 3]
            if ball[0] < -7 or ball[0] > 7 or abs(ball[1]) > 4.5:
                if ball[0] > 7 and abs(ball[1]) < 1.3:
                    self.goals += 1
                else:
                    self.outs += 1
                self.reset_ball()
                self.history[:] = 0
                self.control_steps = 0
                print(f'Goal: {self.goals}, Out: {self.outs}, Fall: {self.falls}', flush=True)
        if self.physics_steps % self.decimation == 0:
            obs = self.observation()
            if self.control_steps == 0:
                self.history[:] = obs
            else:
                self.history[1:] = self.history[:-1].copy()
                self.history[0] = obs
            with torch.inference_mode():
                action = self.policy(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(self.history.copy()).unsqueeze(0))
            self.actions[:] = np.clip(action.squeeze(0).numpy(), -self.cfg['normalization']['clip_actions'], self.cfg['normalization']['clip_actions'])
            self.targets[:] = self.default_pos + self.action_scale * self.actions
            self.control_steps += 1
        torque = self.kp * (self.targets - self.data.qpos[self.pos_addrs]) - self.kd * self.data.qvel[self.vel_addrs]
        self.data.ctrl[:] = np.clip(torque, -self.effort_limits, self.effort_limits)
        mujoco.mj_step(self.model, self.data)
        self.physics_steps += 1
        self.total_steps += 1


def run(checkpoint=None, headless=False, steps=None, seed=42, reset_mode='reference', task='Q1',
        show_frustum=True, frustum_range=2.5):
    sim = SoccerPlayer(checkpoint, seed, reset_mode, task)
    limit = steps if steps is not None else (5000 if headless else None)
    if headless:
        for _ in range(limit):
            sim.step()
    else:
        visible = [show_frustum]
        def key_callback(key):
            if key in (ord('F'), ord('f')):
                visible[0] = not visible[0]
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 5
            viewer.cam.elevation = -20
            count = 0
            while viewer.is_running() and (limit is None or count < limit):
                start = time.time()
                sim.step()
                viewer.cam.lookat[:] = sim.data.sensor('position').data
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if visible[0]:
                        draw_camera_frustum(viewer.user_scn, sim.model, sim.data, sim.cfg, frustum_range)
                viewer.sync()
                count += 1
                remaining = sim.model.opt.timestep - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    print(f'Steps: {sim.total_steps}, Goals: {sim.goals}, Out: {sim.outs}, Falls: {sim.falls}')
    return sim


def view_initial_state(seed=42, reset_mode='reference', show_frustum=True, frustum_range=2.5):
    """Inspect a real Q1 batch reset without advancing any physics steps."""
    from envs.q1 import Q1
    import glfw
    if os.environ.get('DISPLAY') and hasattr(glfw, 'PLATFORM_X11'):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    with open('envs/Q1.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['basic']['device'] = 'cpu'
    cfg['env']['num_envs'] = 1
    cfg['physics']['num_threads'] = 1
    cfg['motion']['reference_state_init'] = reset_mode == 'reference'
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = Q1(cfg)
    env.reset()
    model = env.model
    data = mujoco.MjData(model)
    # Match this simulation's randomized physical parameters as well as its pose.
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'geom_friction'):
        getattr(model, name)[:] = env.batch.expand(name)[0]
    mujoco.mj_setConst(model, data)
    data.qpos[:] = env.qpos[0]
    data.qvel[:] = env.qvel[0]
    mujoco.mj_forward(model, data)
    print(f'Q1 initial state held at t={data.time:.3f}s; reset={reset_mode}, '
          f'base height={data.qpos[env.root_addr + 2]:.3f}m. '
          'Physics is frozen; drag to rotate, scroll to zoom, close the window to exit.', flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.qpos[env.root_addr:env.root_addr + 3] + [0, 0, .1]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if show_frustum:
                    draw_camera_frustum(viewer.user_scn, model, data, cfg, frustum_range)
            viewer.sync()
            time.sleep(1 / 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('Q1', 'G1', 'T1'), default='Q1')
    parser.add_argument('--checkpoint', help='Task checkpoint; defaults to latest in its log directory')
    parser.add_argument('--headless', action='store_true', help='run without a viewer')
    parser.add_argument('--steps', type=int, help='number of MuJoCo physics steps to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-frustum', action='store_true', help='hide the camera frustum (F toggles it during playback)')
    parser.add_argument('--frustum-range', type=float, default=2.5, help='display length in metres; not a sensor range limit')
    parser.add_argument('--initial-state', action='store_true', help='open a frozen Q1 training reset; no checkpoint required')
    parser.add_argument('--reset-mode', choices=('reference', 'default'), default='reference',
                        help='reference uses motion-state resets; default uses the neutral stance')
    args = parser.parse_args()
    if args.frustum_range <= 0:
        parser.error('--frustum-range must be positive')
    if args.initial_state:
        if args.task != 'Q1' or args.headless:
            parser.error('--initial-state requires --task Q1 and a graphical display')
        view_initial_state(args.seed, args.reset_mode, not args.no_frustum, args.frustum_range)
        return
    run(args.checkpoint, args.headless, args.steps, args.seed, args.reset_mode, args.task,
        not args.no_frustum, args.frustum_range)


if __name__ == '__main__':
    main()
