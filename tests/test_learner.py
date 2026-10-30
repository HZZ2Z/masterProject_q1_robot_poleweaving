import mujoco
import numpy as np
import torch
from q1_supermimic.physics import SCENE
from q1_supermimic.observations import MotorObservation
from q1_supermimic.dynamics import Dynamics
from q1_supermimic.tracking import TrackingHead
from q1_supermimic.networks import MotorPolicy, mlp
from q1_supermimic.learner import tracking_update


def test_motor_observation_excludes_ball_and_world_position():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    observe = MotorObservation(model)
    state = torch.tensor(np.r_[model.qpos0, np.zeros(model.nv)], dtype=torch.float32)
    previous = torch.zeros(22)
    changed = state.clone()
    changed[:3] += 7
    changed[29:36] += 10
    changed[model.nq+28:] += 100
    assert observe(state, previous).shape == (72,)
    assert observe.reference(state).shape == (54,)
    torch.testing.assert_close(observe(state, previous), observe(changed, previous))


def test_actual_tracking_update_changes_actor_and_critic_but_not_world():
    torch.manual_seed(41)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    observe, head = MotorObservation(model), TrackingHead(model)
    world = Dynamics(model, widths=(16,))
    policy = MotorPolicy(2*72, 2*54, 4, 22, motor_widths=(16,), visual_widths=(8,))
    critic = mlp(70+2*54, 1, (16,))
    state = torch.tensor(np.r_[model.qpos0, np.zeros(model.nv)], dtype=torch.float32).repeat(2, 1)
    state[:, observe.control.qpos_indices] = observe.control.neutral
    history = observe(state, torch.zeros(2,22)).unsqueeze(1).repeat(1,2,1)
    before_world = {k:v.clone() for k,v in world.state_dict().items()}
    before_actor = {k:v.clone() for k,v in policy.state_dict().items()}
    before_critic = {k:v.clone() for k,v in critic.state_dict().items()}
    result = tracking_update(policy, world, critic, head, observe,
                             torch.optim.Adam(policy.parameters(),lr=2e-4),
                             torch.optim.Adam(critic.parameters(),lr=2e-4),
                             state, history, state.unsqueeze(0).repeat(5,1,1),
                             torch.zeros(2,0), horizon=2,future=2,critic_epochs=1)
    assert np.isfinite(list(result.values())).all()
    assert result['actor_grad_norm'] > 0
    assert all(torch.equal(v,before_world[k]) for k,v in world.state_dict().items())
    assert all(p.grad is None for p in world.parameters())
    assert any(not torch.equal(v,before_actor[k]) for k,v in policy.state_dict().items())
    assert any(not torch.equal(v,before_critic[k]) for k,v in critic.state_dict().items())
    assert all(p.requires_grad for p in world.parameters())
