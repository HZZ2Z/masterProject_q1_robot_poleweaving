# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import csv
import os

import numpy as np
import pinocchio as pin
import torch
from isaacgym.torch_utils import quat_mul, quat_rotate_inverse


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
        for file in sorted(file_list):
            with open(file, "r", encoding="utf-8") as f:
                data = np.asarray(list(csv.reader(f))[1:], dtype=np.float32)
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

        pin_model = pin.buildModelFromUrdf(self.cfg["asset"]["file"])
        pin_data = pin_model.createData()
        if pin_model.nq != self.num_dofs:
            raise ValueError(f"URDF has {pin_model.nq} DoFs, config has {self.num_dofs}")
        for idx in range(self.data.shape[0]):
            dof_pos = pin.neutral(pin_model)
            dof_pos[self.dof_start_index :] = self.data[idx, amp_pos_start:amp_vel_start].cpu().numpy()
            pin.forwardKinematics(pin_model, pin_data, dof_pos)
            pin.updateFramePlacements(pin_model, pin_data)
            for foot_idx, foot_name in enumerate(cfg["asset"]["foot_names"]):
                start = amp_feet_start + 3 * foot_idx
                self.data[idx, start : start + 3] = torch.as_tensor(
                    pin_data.oMf[pin_model.getFrameId(foot_name)].translation,
                    dtype=torch.float32,
                    device=self.device,
                )

    def sample_batch(self, batch_size):
        sampled_idxs = self.valid_idxs[torch.randint(0, self.valid_idxs.shape[0], (batch_size,), device=self.device)]
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
