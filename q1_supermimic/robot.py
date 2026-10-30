"""Q1 actuator contract shared by native validation and Warp collection.

Addresses come from transmission joint IDs, never assumed qpos slices.
The model contains two free joints (robot and ball).
"""
import mujoco
import torch
from torch import nn


class Q1Control(nn.Module):
    control_period = 0.02

    def __init__(self, model):
        super().__init__()
        joints = model.actuator_trnid[:, 0]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
        if len(names) != 22 or len(set(names)) != 22:
            raise ValueError('Expected 22 independently actuated Q1 joints')
        gains = {'hip': (60., 2.), 'knee': (80., 2.5), 'ankle': (60., 2.5),
                 'waist': (40., 1.5), 'shoulder': (20., .8), 'elbow': (20., .8)}
        neutral = {'hip_pitch': -.2, 'knee': .4, 'ankle_pitch': -.2, 'elbow': .8}
        scales = {'hip_pitch': 1.1, 'hip_roll': .95, 'hip_yaw': 1.1, 'knee': .9,
                  'ankle_pitch': .65, 'ankle_roll': .25, 'waist_roll': .15,
                  'waist_yaw': .6, 'shoulder_pitch': .4, 'shoulder_roll': .3,
                  'shoulder_yaw': .4, 'elbow': .4}
        q0, kp, kd, scale = [], [], [], []
        for name in names:
            kind = name.removesuffix('_joint').removeprefix('left_').removeprefix('right_')
            scale.append(scales[kind])
            gain = next(v for k, v in gains.items() if k in name)
            kp.append(gain[0]); kd.append(gain[1])
            q0.append(next((v for k, v in neutral.items() if k in name), 0.))
            if 'shoulder_roll' in name:
                q0[-1] = .15 if name.startswith('left') else -.15
        arrays = dict(qpos_indices=torch.tensor(model.jnt_qposadr[joints], dtype=torch.long),
                      qvel_indices=torch.tensor(model.jnt_dofadr[joints], dtype=torch.long),
                      neutral=torch.tensor(q0), kp=torch.tensor(kp), kd=torch.tensor(kd),
                      lower=torch.tensor(model.jnt_range[joints, 0], dtype=torch.float32),
                      upper=torch.tensor(model.jnt_range[joints, 1], dtype=torch.float32),
                      torque_lower=torch.tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32),
                      torque_upper=torch.tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32),
                      scale=torch.tensor(scale))
        for key, value in arrays.items():
            self.register_buffer(key, value)
        self.names = names
        self.substeps = round(self.control_period / model.opt.timestep)
        if abs(self.substeps * model.opt.timestep - self.control_period) > 1e-9:
            raise ValueError('Control period must be an integer number of physics steps')

    def target(self, action):
        return (self.neutral + self.scale * action).clamp(self.lower, self.upper)

    def forward(self, qpos, qvel, action):
        torque = self.kp * (self.target(action) - qpos[..., self.qpos_indices])
        torque = torque - self.kd * qvel[..., self.qvel_indices]
        return torque.clamp(self.torque_lower, self.torque_upper)


def check_stance():
    """Five-second Warp control check; does not optimize any policy."""
    import json
    from .physics import WarpPhysics

    torch.cuda.set_per_process_memory_fraction(.4)
    physics = WarpPhysics(16)
    control = Q1Control(physics.native_model).to(physics.device)
    position = torch.tensor(physics.native_model.qpos0, device=physics.device,
                            dtype=torch.float32).repeat(16, 1)
    position[:, control.qpos_indices] = control.neutral
    physics.reset(position, torch.zeros_like(physics.qvel))
    action = torch.zeros((16, 22), device=physics.device)
    heights = []
    for _ in range(250):
        for _ in range(control.substeps):
            physics.step(control(physics.qpos, physics.qvel, action))
        heights.append(physics.qpos[:, 2].clone())
    physics.check()
    height = torch.stack(heights)
    result = dict(worlds=16, seconds=5, minimum_height=height.min().item(),
                  final_height=height[-1].mean().item(),
                  surviving=int((height.min(0).values > .3).sum()))
    print(json.dumps(result))
    if result['surviving'] != result['worlds']:
        raise RuntimeError('Neutral stance failed in Warp')


if __name__ == '__main__':
    check_stance()
