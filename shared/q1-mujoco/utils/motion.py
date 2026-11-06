# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import csv
import os

import numpy as np
import xml.etree.ElementTree as ET
import torch
from utils.torch_math import quat_mul, quat_rotate_inverse
from utils.kinematics import ResetKinematics


class Motion:

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.dof_start_index = cfg["motion"].get("dof_start_index", 2)
        self.num_dofs = cfg["env"]["num_actions"]
        self.num_motion_dofs = self.num_dofs - self.dof_start_index
        expected_amp_obs = 15 + 2 * self.num_motion_dofs
        if cfg["env"]["num_amp_obs"] != expected_amp_obs:
            raise ValueError(
                f"num_amp_obs must be {expected_amp_obs} for {self.num_motion_dofs} mocap DoFs, "
                f"got {cfg['env']['num_amp_obs']}"
            )

        file_list = []
        motion_path = cfg["motion"]["motion_path"]
        if os.path.isdir(motion_path):
            file_list.extend(
                os.path.join(motion_path, file)
                for file in os.listdir(motion_path)
                if not file.startswith(".")
                and file.lower().endswith(".csv")
                and os.path.isfile(os.path.join(motion_path, file))
            )
        elif os.path.isfile(motion_path):
            file_list.append(motion_path)
        else:
            raise FileNotFoundError(f"Motion path does not exist: {motion_path}")
        if not file_list:
            raise FileNotFoundError(f"No CSV motion files found in: {motion_path}")

        origin_data = []
        valid_idxs = []
        start_idx = 0
        expected_columns = 13 + 2 * self.num_motion_dofs
        names = cfg["asset"].get("dof_names")
        expected_header = None
        if names:
            names = names[self.dof_start_index:]
            expected_header = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                               'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z']
            expected_header += ['q_' + n for n in names] + ['dq_' + n for n in names]
        for file in sorted(file_list):
            with open(file, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if expected_header is not None and rows[0] != expected_header:
                raise ValueError(f"{file} joint columns differ from the configured joint order")
            data = np.asarray(rows[1:], dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f"Non-finite motion data in {file}")
            if data.ndim != 2 or data.shape[1] != expected_columns:
                raise ValueError(f"{file} must have {expected_columns} columns, got {data.shape}")
            if data.shape[0] < 2:
                raise ValueError(f"{file} must contain at least two motion frames")
            motion_data = torch.from_numpy(data).to(self.device)
            origin_data.append(motion_data)
            end_idx = start_idx + motion_data.shape[0]
            valid_idxs.append(torch.arange(start_idx, end_idx - 1, device=self.device))
            start_idx = end_idx
        origin_data = torch.cat(origin_data, dim=0)
        self.valid_idxs = torch.cat(valid_idxs, dim=0)

        self.base_height = origin_data[:, 2].clone()
        self.base_quat = origin_data[:, 3:7].clone()
        pos_start = 13
        vel_start = pos_start + self.num_motion_dofs
        amp_pos_start = 9
        amp_vel_start = amp_pos_start + self.num_motion_dofs
        amp_feet_start = amp_vel_start + self.num_motion_dofs
        self.data = torch.zeros(origin_data.shape[0], self.cfg["env"]["num_amp_obs"], dtype=torch.float32, device=self.device)
        self.data[:, 0:3] = quat_rotate_inverse(
            origin_data[:, 3:7], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).unsqueeze(0).expand(origin_data.shape[0], -1)
        )
        self.data[:, 3:9] = origin_data[:, 7:13]
        self.data[:, amp_pos_start:amp_vel_start] = origin_data[:, pos_start:vel_start]
        self.data[:, amp_vel_start:amp_feet_start] = origin_data[:, vel_start : vel_start + self.num_motion_dofs]
        speed_squared = self.data[:, 3:9].square().sum(-1)
        limit = cfg['rewards']['terminate_vel']
        # Keep consecutive valid pairs; do not splice across rejected frames.
        self.valid_idxs = self.valid_idxs[(speed_squared[self.valid_idxs] < limit) &
                                          (speed_squared[self.valid_idxs + 1] < limit)]
        self.reset_idxs = self.valid_idxs[speed_squared[self.valid_idxs] < .5 * limit]
        if not len(self.reset_idxs):
            raise ValueError('No reference frames are safely below the base velocity termination limit')

        joints = [j for j in ET.parse(cfg["asset"]["file"]).getroot().findall('joint') if j.get('type') != 'fixed']
        dof_names = cfg["asset"].get("dof_names", [j.get('name') for j in joints])
        if len(joints) != self.num_dofs:
            raise ValueError(f"URDF has {len(joints)} DoFs, config has {self.num_dofs}")
        fk = ResetKinematics(cfg["asset"]["file"], dof_names, cfg["asset"]["foot_names"], self.device)
        q = torch.zeros((len(origin_data), self.num_dofs), device=self.device)
        q[:, self.dof_start_index:] = self.data[:, amp_pos_start:amp_vel_start]
        frames = fk(q)
        for foot_idx, name in enumerate(cfg["asset"]["foot_names"]):
            start = amp_feet_start + 3 * foot_idx
            self.data[:, start:start + 3] = frames[name][0]

    def sample_batch(self, batch_size):
        sampled_idxs = self.reset_idxs[torch.randint(0, self.reset_idxs.shape[0], (batch_size,), device=self.device)]
        yaw_offset = torch.rand(batch_size, dtype=torch.float32, device=self.device) * (2 * np.pi)
        yaw_quat = torch.stack(
            [torch.zeros_like(yaw_offset), torch.zeros_like(yaw_offset), torch.sin(yaw_offset / 2), torch.cos(yaw_offset / 2)], dim=1
        )
        return {
            "base_height": self.base_height[sampled_idxs],
            "base_quat": quat_mul(yaw_quat, self.base_quat[sampled_idxs]),
            "base_lin_vel": self.data[sampled_idxs, 3:6],
            "base_ang_vel": self.data[sampled_idxs, 6:9],
            "dof_pos": self.data[sampled_idxs, 9 : 9 + self.num_motion_dofs],
            "dof_vel": self.data[
                sampled_idxs, 9 + self.num_motion_dofs : 9 + 2 * self.num_motion_dofs
            ],
        }

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        amp_vel_start = 9 + self.num_motion_dofs
        amp_feet_start = amp_vel_start + self.num_motion_dofs
        for _ in range(num_mini_batch):
            sampled_idxs = self.valid_idxs[torch.randint(0, self.valid_idxs.shape[0], (mini_batch_size,), device=self.device)]
            state = self.data[sampled_idxs].clone()
            next_state = self.data[sampled_idxs + 1].clone()
            state[:, amp_vel_start:amp_feet_start] *= self.cfg["normalization"]["dof_vel"]
            next_state[:, amp_vel_start:amp_feet_start] *= self.cfg["normalization"]["dof_vel"]
            yield state, next_state
