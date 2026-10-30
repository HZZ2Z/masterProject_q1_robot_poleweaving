"""Differentiable body poses from compiled MuJoCo model geometry (wxyz)."""
import mujoco
import torch
from torch import nn
from .dynamics import multiply, integrate


def rotate(quaternion, vector):
    vector = torch.broadcast_to(vector, (*quaternion.shape[:-1], 3))
    uv = torch.linalg.cross(quaternion[..., 1:], vector)
    return vector + 2 * (quaternion[..., :1] * uv + torch.linalg.cross(quaternion[..., 1:], uv))


class Kinematics(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for name in ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'qpos0'):
            self.register_buffer(name, torch.tensor(getattr(model, name), dtype=torch.float64))
        if any(model.body_jntnum > 1):
            raise ValueError('Q1 kinematics supports at most one joint per body')
        if any(int(k) not in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE)) for k in model.jnt_type):
            raise ValueError('Only free and hinge joints are supported')

    def forward(self, qpos):
        pos0 = qpos.new_zeros((*qpos.shape[:-1], 3))
        quat0 = qpos.new_zeros((*qpos.shape[:-1], 4))
        quat0[..., 0] = 1
        positions, orientations = [pos0], [quat0]
        for body in range(1, self.model.nbody):
            parent = int(self.model.body_parentid[body])
            parent_q, parent_p = orientations[parent], positions[parent]
            position = parent_p + rotate(parent_q, self.body_pos[body].to(qpos))
            orientation = multiply(parent_q, self.body_quat[body].to(qpos).expand_as(parent_q))
            if self.model.body_jntnum[body]:
                joint = int(self.model.body_jntadr[body])
                address = int(self.model.jnt_qposadr[joint])
                if int(self.model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    position = qpos[..., address:address+3]
                    orientation = torch.nn.functional.normalize(qpos[..., address+3:address+7], dim=-1)
                else:
                    angle = qpos[..., address:address+1] - self.qpos0[address].to(qpos)
                    axis = self.jnt_axis[joint].to(qpos)
                    delta = torch.cat(((angle / 2).cos(), (angle / 2).sin() * axis), -1)
                    anchor = self.jnt_pos[joint].to(qpos)
                    fixed_anchor = position + rotate(orientation, anchor)
                    orientation = multiply(orientation, delta)
                    position = fixed_anchor - rotate(orientation, anchor)
            positions.append(position)
            orientations.append(orientation)
        return torch.stack(positions, -2), torch.stack(orientations, -2)

    def linear_velocity(self, qpos, qvel):
        return self.velocities(qpos, qvel)[0]

    def state_features(self, qpos, qvel):
        """Propagate body-origin twists analytically, including joint anchors.

        This avoids a nested forward-mode AD pass during every imagined step;
        all operations retain gradients in both configuration and velocity.
        """
        zero = qpos.new_zeros((*qpos.shape[:-1], 3))
        identity = qpos.new_zeros((*qpos.shape[:-1], 4))
        identity[..., 0] = 1
        positions, orientations = [zero], [identity]
        velocities, angulars = [zero], [zero]
        for body in range(1, self.model.nbody):
            parent = int(self.model.body_parentid[body])
            pq, pp = orientations[parent], positions[parent]
            pv, pw = velocities[parent], angulars[parent]
            offset = rotate(pq, self.body_pos[body].to(qpos))
            position = pp + offset
            orientation = multiply(pq, self.body_quat[body].to(qpos).expand_as(pq))
            velocity = pv + torch.linalg.cross(pw, offset)
            angular = pw
            if self.model.body_jntnum[body]:
                joint = int(self.model.body_jntadr[body])
                address = int(self.model.jnt_qposadr[joint])
                speed = int(self.model.jnt_dofadr[joint])
                if int(self.model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    position = qpos[..., address:address+3]
                    orientation = torch.nn.functional.normalize(qpos[..., address+3:address+7], dim=-1)
                    velocity = qvel[..., speed:speed+3]
                    angular = rotate(orientation, qvel[..., speed+3:speed+6])
                else:
                    angle = qpos[..., address:address+1] - self.qpos0[address].to(qpos)
                    axis = self.jnt_axis[joint].to(qpos)
                    anchor = self.jnt_pos[joint].to(qpos)
                    old_anchor = rotate(orientation, anchor)
                    fixed_anchor = position + old_anchor
                    anchor_velocity = velocity + torch.linalg.cross(pw, old_anchor)
                    angular = pw + rotate(orientation, axis) * qvel[..., speed:speed+1]
                    delta = torch.cat(((angle / 2).cos(), (angle / 2).sin() * axis), -1)
                    orientation = multiply(orientation, delta)
                    new_anchor = rotate(orientation, anchor)
                    position = fixed_anchor - new_anchor
                    velocity = anchor_velocity - torch.linalg.cross(angular, new_anchor)
            positions.append(position)
            orientations.append(orientation)
            velocities.append(velocity)
            angulars.append(angular)
        return tuple(torch.stack(values, -2) for values in (positions,orientations,velocities,angulars))

    def velocities(self, qpos, qvel):
        return self.state_features(qpos, qvel)[2:]
