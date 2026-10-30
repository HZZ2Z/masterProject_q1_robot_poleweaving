import mujoco
import numpy as np
import torch
from q1_supermimic.physics import SCENE
from q1_supermimic.tracking import TrackingHead


def test_tracking_prefers_reference_and_penalizes_fall_with_finite_gradient():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    head = TrackingHead(model)
    state = torch.tensor(np.r_[model.qpos0, np.zeros(model.nv)], dtype=torch.float64)
    state[head.control.qpos_indices] = head.control.neutral.double()
    action = torch.zeros(22, dtype=torch.float64)
    reward, done, probability = head(state, state, action, action)
    assert not done and probability < .001
    perturbed = state.clone()
    perturbed[2] -= .3
    perturbed.requires_grad_()
    bad_reward, bad_done, bad_probability = head(perturbed, state, action, action)
    assert bad_done and bad_probability > .9 and bad_reward < reward
    (bad_reward - bad_probability).backward()
    assert torch.isfinite(perturbed.grad).all()
    assert perturbed.grad[2] > 0
