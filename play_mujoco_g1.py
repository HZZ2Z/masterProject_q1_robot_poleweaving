# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run a trained 13-DoF G1 soccer policy in MuJoCo.

The scene adds the ball, field marks, and policy sensors to the same G1 model
used by the training configuration. Run from the repository root.
"""

import argparse
import time
from pathlib import Path

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


def latest_g1_checkpoint():
    paths = list(Path('logs').glob('g1_13dof_*/logs/*/nn/model_*.pth'))
    if not paths:
        raise FileNotFoundError('No G1 13-DoF checkpoint found; pass --checkpoint')
    return max(paths, key=lambda path: path.stat().st_mtime)


class G1Soccer:
    def __init__(self, checkpoint=None, seed=42, reset_mode='reference'):
        with open('envs/G1.yaml', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.rng = np.random.default_rng(seed)
        self.reset_mode = reset_mode
        self.reference_frames = None
        if reset_mode == 'reference':
            files = sorted(Path(self.cfg['motion']['motion_path']).glob('*.csv'))
            if not files:
                raise FileNotFoundError('G1 reference motion CSV files are missing')
            self.reference_frames = np.concatenate([
                np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)[:-1]
                for path in files
            ])
        path = Path(checkpoint) if checkpoint and checkpoint != '-1' else latest_g1_checkpoint()
        self.policy = ActorCritic(self.cfg)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        self.policy.load_state_dict(saved['model'], strict=True)
        self.policy.eval()
        print(f'Loading G1 policy from {path}', flush=True)

        self.model = mujoco.MjModel.from_xml_path(self.cfg['asset']['mujoco_file'])
        self.model.opt.timestep = self.cfg['sim']['dt']
        self.data = mujoco.MjData(self.model)
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        if names != self.cfg['asset']['dof_names']:
            raise ValueError(f'MuJoCo actuator order differs from G1 training: {names}')
        if self.model.nu != self.cfg['env']['num_actions']:
            raise ValueError('MuJoCo actuator count differs from the G1 policy')
        joints = self.model.actuator_trnid[:, 0]
        self.pos_addrs = self.model.jnt_qposadr[joints]
        self.vel_addrs = self.model.jnt_dofadr[joints]
        self.root_addr = self.model.jnt_qposadr[self.model.joint('floating_base_joint').id]
        self.ball_addr = self.model.jnt_qposadr[self.model.joint('ball_free').id]
        self.ball_vel_addr = self.model.jnt_dofadr[self.model.joint('ball_free').id]
        self.effort_limits = np.abs(self.model.jnt_actfrcrange[joints, 1])
        if np.any(self.effort_limits <= 0):
            raise ValueError('G1 model is missing joint effort limits')

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
        self.reset_robot()

    def reset_ball(self):
        a = self.ball_addr
        self.data.qpos[a:a + 7] = [self.rng.uniform(-7, 7), self.rng.uniform(-4.5, 4.5), 0.1, 1, 0, 0, 0]
        self.data.qvel[self.ball_vel_addr:self.ball_vel_addr + 6] = 0
        self.ball_history[:] = 0
        self.ball_delay = int(np.clip(np.rint(self.rng.normal(6, 1)), 0, len(self.ball_history) - 1))
        mujoco.mj_forward(self.model, self.data)

    def reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        a = self.root_addr
        yaw = self.rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([0., 0., np.sin(yaw / 2), np.cos(yaw / 2)])
        base_xy = [self.rng.uniform(-7, 7), self.rng.uniform(-4.5, 4.5)]
        self.data.qvel[:] = 0
        if self.reset_mode == 'reference':
            frame = self.reference_frames[self.rng.integers(len(self.reference_frames))]
            base_quat = quat_mul(yaw_quat, frame[3:7])
            self.data.qpos[a:a + 7] = [*base_xy, frame[2] + 0.01,
                                         base_quat[3], base_quat[0], base_quat[1], base_quat[2]]
            self.data.qpos[self.pos_addrs] = frame[13:26]
            base_vel_addr = self.model.jnt_dofadr[self.model.joint('floating_base_joint').id]
            self.data.qvel[base_vel_addr:base_vel_addr + 3] = quat_rotate(base_quat, frame[7:10])
            self.data.qvel[base_vel_addr + 3:base_vel_addr + 6] = frame[10:13]
            self.data.qvel[self.vel_addrs] = frame[26:39]
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
        self.reset_ball()

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
        obs[13:26] = (self.data.qpos[self.pos_addrs] - self.default_pos) * norm['dof_pos']
        obs[26:39] = self.data.qvel[self.vel_addrs] * norm['dof_vel']
        obs[39:52] = self.actions
        return obs

    def step(self):
        if self.data.sensor('position').data[2] < self.cfg['rewards']['terminate_height']:
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


def run(checkpoint=None, headless=False, steps=None, seed=42, reset_mode='reference'):
    sim = G1Soccer(checkpoint, seed, reset_mode)
    limit = steps if steps is not None else (5000 if headless else None)
    if headless:
        for _ in range(limit):
            sim.step()
    else:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = 5
            viewer.cam.elevation = -20
            count = 0
            while viewer.is_running() and (limit is None or count < limit):
                start = time.time()
                sim.step()
                viewer.cam.lookat[:] = sim.data.sensor('position').data
                viewer.sync()
                count += 1
                remaining = sim.model.opt.timestep - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    print(f'Steps: {sim.total_steps}, Goals: {sim.goals}, Out: {sim.outs}, Falls: {sim.falls}')
    return sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('G1',), default='G1')
    parser.add_argument('--checkpoint', help='G1 13-DoF checkpoint; defaults to the latest G1 checkpoint')
    parser.add_argument('--headless', action='store_true', help='run without a viewer')
    parser.add_argument('--steps', type=int, help='number of MuJoCo physics steps to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reset-mode', choices=('reference', 'default'), default='reference',
                        help='reference matches Isaac Gym motion-state resets')
    args = parser.parse_args()
    run(args.checkpoint, args.headless, args.steps, args.seed, args.reset_mode)


if __name__ == '__main__':
    main()
