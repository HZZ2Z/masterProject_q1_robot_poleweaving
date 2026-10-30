import mujoco
import numpy as np
import torch

from q1_supermimic.physics import SCENE
from q1_supermimic.robot import Q1Control


def test_joint_mapping_excludes_ball_and_pd_preserves_action_gradient():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    control = Q1Control(model)
    assert control.qpos_indices.max() < model.jnt_qposadr[-1]
    state = torch.tensor(model.qpos0, dtype=torch.float32)
    state[control.qpos_indices] = control.neutral
    velocity = torch.zeros(model.nv)
    action = torch.zeros(model.nu, requires_grad=True)
    torque = control(state, velocity, action)
    torch.testing.assert_close(torque, torch.zeros(model.nu))
    torque.sum().backward()
    assert (action.grad > 0).all()
    saturated = control(state, velocity, torch.full((model.nu,), 100.))
    assert (saturated <= control.torque_upper).all()
    assert control.substeps == 10


def test_native_neutral_stance_survives_five_seconds():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    control = Q1Control(model)
    data = mujoco.MjData(model)
    data.qpos[control.qpos_indices.numpy()] = control.neutral.numpy()
    mujoco.mj_forward(model, data)
    # Fixed controller baseline; no policy, training, randomization or ball target.
    for _ in range(round(5 / model.opt.timestep)):
        position = data.qpos[control.qpos_indices.numpy()]
        velocity = data.qvel[control.qvel_indices.numpy()]
        data.ctrl[:] = np.clip(control.kp.numpy() * (control.neutral.numpy() - position)
                               - control.kd.numpy() * velocity,
                               control.torque_lower.numpy(), control.torque_upper.numpy())
        mujoco.mj_step(model, data)
        assert data.qpos[2] > .3
    assert np.isfinite(data.qpos).all()
