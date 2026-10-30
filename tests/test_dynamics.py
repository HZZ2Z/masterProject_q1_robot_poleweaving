import mujoco
import numpy as np
import torch
from q1_supermimic.physics import SCENE
from q1_supermimic.dynamics import Dynamics, integrate


def test_manifold_integration_matches_mujoco_and_zero_rotation_has_gradient():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    rng = np.random.default_rng(16)
    for _ in range(8):
        displacement = rng.normal(size=model.nv) * .1
        expected = model.qpos0.copy()
        mujoco.mj_integratePos(model, expected, displacement, 1.)
        actual = integrate(model, torch.tensor(model.qpos0), torch.tensor(displacement))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    displacement = torch.zeros(model.nv, dtype=torch.float64, requires_grad=True)
    integrate(model, torch.tensor(model.qpos0), displacement).sum().backward()
    assert torch.isfinite(displacement.grad).all()
    assert displacement.grad.abs().sum() > 0


def test_quaternion_sign_invariance_and_autoregressive_action_gradient():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    world = Dynamics(model, widths=(32,))
    state = torch.tensor(np.r_[model.qpos0, np.zeros(model.nv)], dtype=torch.float32).repeat(3, 1)
    flipped = state.clone()
    flipped[:, 3:7] *= -1
    flipped[:, 32:36] *= -1
    assert world.error(flipped, state).item() == 0
    actions = torch.zeros(8, 3, model.nu, requires_grad=True)
    context = torch.zeros(8, 3, 0)
    target = state.repeat(8, 1, 1)
    loss = world.window_loss(dict(state=target, action=actions, context=context, next_state=target))
    loss.backward()
    assert torch.isfinite(actions.grad).all()
    assert (actions.grad.abs().sum((1, 2)) > 0).all()
