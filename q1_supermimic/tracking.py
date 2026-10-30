"""Analytic reference-tracking head; training-only reference interface.

This head never supplies reference states to the final visual soccer actor.
Thresholds and reward widths follow the paper's tracking formulation, while
body selection is an explicit Q1 adaptation. Body velocities use body origins.
"""
import mujoco
import torch
from torch import nn
from .kinematics import Kinematics, rotate
from .robot import Q1Control


def rotation_error_squared(a, b):
    # Squared shortest geodesic angle; stable at identical orientations.
    dot = (a * b).sum(-1).abs().clamp(max=1.)
    squared_sine = (1 - dot.square()).clamp_min(0.)
    angle = 2 * torch.atan2(squared_sine.clamp_min(1e-12).sqrt(), dot)
    return torch.where(squared_sine < 1e-8, 4 * squared_sine, angle.square())


class TrackingHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fk = Kinematics(model)
        self.control = Q1Control(model)
        self.nq = model.nq
        self.root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        names = ['pelvis', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link',
                 'left_elbow_link', 'right_elbow_link']
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
        if min(ids) < 0:
            raise ValueError('Missing tracking body')
        self.register_buffer('bodies', torch.tensor(ids, dtype=torch.long))

    def features(self, state):
        position, rotation = self.fk(state[..., :self.nq])
        velocity, angular = self.fk.velocities(state[..., :self.nq], state[..., self.nq:])
        return position, rotation, velocity, angular

    def forward(self, state, reference, action, previous_action):
        p, q, v, w = self.features(state)
        rp, rq, rv, rw = self.features(reference)
        ids = self.bodies
        body_position = ((p[..., ids, :] - rp[..., ids, :]).square().sum(-1).mean(-1) / .3**2).neg().exp()
        body_rotation = (rotation_error_squared(q[..., ids, :], rq[..., ids, :]).mean(-1) / .4**2).neg().exp()
        body_velocity = ((v[..., ids, :] - rv[..., ids, :]).square().sum(-1).mean(-1)).neg().exp()
        root_position = ((p[..., self.root, :] - rp[..., self.root, :]).square().sum(-1) / .3**2).neg().exp()
        root_rotation = (rotation_error_squared(q[..., self.root, :], rq[..., self.root, :]) / .4**2).neg().exp()
        joints = state[..., self.control.qpos_indices]
        limits = (joints - self.control.upper).clamp_min(0) + (self.control.lower - joints).clamp_min(0)
        body_angular = ((w[..., ids, :] - rw[..., ids, :]).square().sum(-1).mean(-1) / 3.142**2).neg().exp()
        reward = body_position + body_rotation + body_velocity + body_angular + .5 * (root_position + root_rotation)
        reward = reward - .1 * (action - previous_action).square().sum(-1) - 2 * limits.sum(-1)
        height_error = (p[..., self.root, 2] - rp[..., self.root, 2]).abs()
        gravity = torch.tensor([0., 0., -1.], device=state.device, dtype=state.dtype)
        # z component of projected gravity equals world z component of local z.
        tilt_error = (rotate(q[..., self.root, :], gravity)[..., 2]
                      - rotate(rq[..., self.root, :], gravity)[..., 2]).abs()
        terminated = (height_error > .25) | (tilt_error > .8)
        height_survival = torch.sigmoid((.25 - height_error) / .02)
        tilt_survival = torch.sigmoid((.8 - tilt_error) / .05)
        soft_terminated = 1 - height_survival * tilt_survival
        return reward * .02, terminated, soft_terminated
