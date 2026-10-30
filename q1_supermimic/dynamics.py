"""Differentiable Q1 physical transition, independent of task observations.

Predict both integrated displacement and endpoint velocity: ten physics substeps
cannot be reconstructed by multiplying endpoint velocity by the control period.
Free-joint rotations use MuJoCo's local angular velocity convention (wxyz).
"""
import mujoco
import torch
from torch import nn
from .networks import mlp


def multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def integrate(model, position, displacement):
    parts = []
    for j in range(model.njnt):
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        kind = int(model.jnt_type[j])
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            parts.append(position[..., q:q+3] + displacement[..., v:v+3])
            q += 3; v += 3
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            angle = displacement[..., v:v+3]
            norm = angle.norm(dim=-1, keepdim=True)
            # sinc supplies a finite derivative at zero rotation.
            delta = torch.cat(((norm / 2).cos(), .5 * torch.sinc(norm / (2 * torch.pi)) * angle), -1)
            quat = multiply(position[..., q:q+4], delta)
            parts.append(torch.nn.functional.normalize(quat, dim=-1))
        else:
            parts.append(position[..., q:q+1] + displacement[..., v:v+1])
    return torch.cat(parts, -1)


class Dynamics(nn.Module):
    """State = [qpos, qvel]; context is explicit and fixed per episode.

    No learned observation/reward shortcuts. Task-specific analytic heads and
    history belong outside this physical model. Scales never clip physical state.
    """
    def __init__(self, model, context_dim=0, widths=(2048, 2048, 2048)):
        super().__init__()
        self.model = model
        self.nq, self.nv = model.nq, model.nv
        self.network = mlp(model.nq + model.nv + model.nu + context_dim, 2 * model.nv, widths)
        self.register_buffer('center', torch.zeros(model.nq + model.nv))
        self.register_buffer('scale', torch.ones(model.nq + model.nv))
        self.register_buffer('output_scale', torch.ones(2 * model.nv))

    @torch.no_grad()
    def set_scales(self, states):
        flat = states.reshape(-1, self.nq + self.nv)
        self.center.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0).clamp_min(.1))
        velocity = flat[:, self.nq:].std(0).clamp_min(.1)
        self.output_scale.copy_(torch.cat((.02 * velocity, velocity)))

    def forward(self, state, action, context):
        inputs = torch.cat(((state - self.center) / self.scale, action, context), -1)
        delta, velocity_delta = (self.network(inputs) * self.output_scale).split(self.nv, -1)
        position = integrate(self.model, state[..., :self.nq], delta)
        return torch.cat((position, state[..., self.nq:] + velocity_delta), -1)

    def error(self, predicted, actual):
        # Quaternion sign is physically irrelevant; align target representation.
        target = actual.clone()
        for j in range(self.model.njnt):
            kind = int(self.model.jnt_type[j])
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                q = int(self.model.jnt_qposadr[j]) + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
                sign = torch.where((predicted[..., q:q+4] * actual[..., q:q+4]).sum(-1, keepdim=True) < 0, -1., 1.)
                target[..., q:q+4] = actual[..., q:q+4] * sign
        return ((predicted - target) / self.scale).square().mean()

    def window_loss(self, batch):
        state = batch['state'][0]
        errors = []
        for action, context, target in zip(batch['action'], batch['context'], batch['next_state']):
            state = self(state, action, context)
            errors.append(self.error(state, target))
        return torch.stack(errors).mean()
