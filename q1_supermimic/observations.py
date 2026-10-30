"""Sensor-only motor observations; reference features are training-only goals."""
import torch
from torch import nn
from .robot import Q1Control
from .kinematics import rotate


class MotorObservation(nn.Module):
    dimension = 72
    reference_dimension = 54

    def __init__(self, model):
        super().__init__()
        self.control = Q1Control(model)
        self.nq = model.nq
        # The Q1 pelvis is the first free joint. Assert rather than silently
        # accept another robot layout with incompatible IMU interpretation.
        if int(model.jnt_qposadr[0]) != 0 or int(model.jnt_dofadr[0]) != 0 or int(model.jnt_type[0]) != 0:
            raise ValueError('Expected Q1 pelvis free joint at state origin')

    def gravity(self, state):
        quat = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        inverse = torch.cat((quat[..., :1], -quat[..., 1:]), -1)
        gravity = state.new_tensor([0., 0., -1.])
        return rotate(inverse, gravity)

    def forward(self, state, previous_action):
        return torch.cat((state[..., self.control.qpos_indices] - self.control.neutral,
                          state[..., self.nq + self.control.qvel_indices],
                          state[..., self.nq+3:self.nq+6], self.gravity(state), previous_action), -1)

    def reference(self, state):
        quat = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        inverse = torch.cat((quat[..., :1], -quat[..., 1:]), -1)
        velocity = rotate(inverse, state[..., self.nq:self.nq+3])
        return torch.cat((state[..., self.control.qpos_indices] - self.control.neutral,
                          state[..., self.nq + self.control.qvel_indices], self.gravity(state),
                          state[..., 2:3], velocity, state[..., self.nq+3:self.nq+6]), -1)
