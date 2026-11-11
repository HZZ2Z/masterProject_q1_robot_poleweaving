# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from utils.torch_math import get_euler_xyz, quat_mul, quat_rotate, quat_rotate_inverse

from utils.kinematics import ResetKinematics

from .t1 import T1


class G1(T1):
    """Unitree G1 version of the vision-driven soccer task."""

    def __init__(self, cfg):
        super().__init__(cfg)
        frames = cfg["asset"]["foot_names"] + [cfg["asset"]["head_name"]]
        self.reset_kinematics = ResetKinematics(cfg["asset"]["file"], self.dof_names, frames, self.device)
        self.camera_offset = torch.as_tensor(cfg["asset"]["camera_pos"], device=self.device)
        self.camera_rotation = torch.as_tensor(cfg["asset"]["camera_rot"], device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # Gym rigid-body states still describe the previous pose until simulate().
        base_quat = self.base_quat[env_ids]
        self.base_yaw[env_ids] = get_euler_xyz(base_quat)[2]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(base_quat, self.root_states[env_ids, 0, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(base_quat, self.root_states[env_ids, 0, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(base_quat, self.gravity_vec[env_ids])
        dx = self.ball_pos[env_ids, 0] - self.base_pos[env_ids, 0]
        dy = self.ball_pos[env_ids, 1] - self.base_pos[env_ids, 1]
        c, s = torch.cos(self.base_yaw[env_ids]), torch.sin(self.base_yaw[env_ids])
        self.relative_ball_pos[env_ids, 0] = c * dx + s * dy
        self.relative_ball_pos[env_ids, 1] = -s * dx + c * dy
        dx, dy = 7 - self.base_pos[env_ids, 0], -self.base_pos[env_ids, 1]
        self.relative_goal_pos[env_ids, 0] = c * dx + s * dy
        self.relative_goal_pos[env_ids, 1] = -s * dx + c * dy

        frames = self.reset_kinematics(self.dof_pos[env_ids])
        for foot_idx, name in enumerate(self.cfg["asset"]["foot_names"]):
            pos, quat = frames[name]
            self.feet_pos[env_ids, foot_idx] = self.base_pos[env_ids] + quat_rotate(base_quat, pos)
            self.feet_quat[env_ids, foot_idx] = quat_mul(base_quat, quat)
        self.last_feet_pos[env_ids] = self.feet_pos[env_ids]
        yaw = get_euler_xyz(self.feet_quat[env_ids].reshape(-1, 4))[2]
        self.feet_yaw[env_ids] = (yaw.reshape(len(env_ids), -1) + torch.pi) % (2 * torch.pi) - torch.pi
        self.last_feet_yaw[env_ids] = self.feet_yaw[env_ids]

        head_pos, head_quat = frames[self.cfg["asset"]["head_name"]]
        mount_quat = quat_mul(base_quat, head_quat)
        self.camera_pos[env_ids] = (
            self.base_pos[env_ids] + quat_rotate(base_quat, head_pos)
            + quat_rotate(mount_quat, self.camera_offset.expand(len(env_ids), -1))
        )
        self.head_quat[env_ids] = quat_mul(mount_quat, self.camera_rotation.expand(len(env_ids), -1))

    def _compute_amp_obs(self):
        return torch.cat(
            (
                self.projected_gravity,
                self.base_lin_vel,
                self.base_ang_vel,
                self.dof_pos,
                self.dof_vel * self.cfg["normalization"]["dof_vel"],
                quat_rotate_inverse(self.base_quat, self.feet_pos[:, 0] - self.base_pos),
                quat_rotate_inverse(self.base_quat, self.feet_pos[:, 1] - self.base_pos),
            ),
            dim=-1,
        )

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    @staticmethod
    def _mirror_joints(values):
        # URDF order: left leg, right leg, waist yaw.
        permutation = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12]
        signs = values.new_tensor(
            [1, -1, -1, 1, 1, -1] * 2
            + [-1]
        )
        return values[..., permutation] * signs

    @staticmethod
    def mirror_obs(obs):
        mirrored = obs.clone()
        mirrored[..., [1, 3, 5, 7, 10, 12]] *= -1
        mirrored[..., 13:26] = G1._mirror_joints(obs[..., 13:26])
        mirrored[..., 26:39] = G1._mirror_joints(obs[..., 26:39])
        mirrored[..., 39:52] = G1._mirror_joints(obs[..., 39:52])
        return mirrored

    @staticmethod
    def mirror_act(act):
        return G1._mirror_joints(act)
