"""Real-data critic targets, matched horizons and bounded accepted actor updates."""

from types import SimpleNamespace

import pytest
import torch

from q1_soccer.algorithm import SoccerSuperMimic
from q1_soccer.train import model_validation
from q1_soccer.trust import PolicyTrust


def test_trust_backtracks_and_actor_rollback_keeps_learned_critic():
    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    algorithm = SimpleNamespace(
        actor=actor,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-6),
        critic=critic,
    )
    trust = PolicyTrust(algorithm, rms_limit=0.01, max_limit=0.02)
    history = torch.ones(16, 1)
    before = actor(history).detach()
    previous = trust.snapshot()
    with torch.no_grad():
        actor.weight.add_(0.4)
    result = trust.constrain(previous, history, before)
    assert result["trust_passed"] and 0 < result["step_fraction"] < 1
    assert 0 < result["action_delta_rms"] <= 0.01
    trust.accept()
    retained = actor.weight.detach().clone()
    with torch.no_grad():
        actor.weight.add_(1.0)
        critic.weight.fill_(42.0)
    trust.reject()
    torch.testing.assert_close(actor.weight, retained, rtol=0, atol=0)
    assert float(critic.weight.detach()) == 42.0
    assert trust.accepted_updates == 1 and trust.consecutive_rejections == 1
    assert algorithm.actor_optimizer.param_groups[0]["lr"] == pytest.approx(5e-7)


def test_model_gate_uses_the_actual_policy_horizon():
    state = torch.zeros(2, 109)
    state[:, [3, 82, 95, 105]] = 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))
            self.scale = torch.ones(109)

        def forward(self, state, action, context):
            return state

    class Replay:
        def sample(self, batch_size, window, device):
            assert window == 16
            return SimpleNamespace(
                states=state[:, None].expand(-1, window, -1),
                next_states=state[:, None].expand(-1, window, -1),
                actions=torch.zeros(2, window, 22),
                context=torch.zeros(2, 62),
            )

    metrics = model_validation(SimpleNamespace(model=Model(), horizon=16), Replay(), 2)
    assert metrics["model_validation_steps"] == 16
    assert metrics["world_rmse"] == 0 and metrics["joint_position_rmse"] == 0
    assert metrics["base_orientation_rmse"] == 0


def test_critic_labels_are_real_finite_episode_returns_without_bootstrap():
    algorithm = SoccerSuperMimic.__new__(SoccerSuperMimic)
    algorithm.gamma = 0.99
    episode = {"states": torch.zeros(4, 109), "actions": torch.zeros(3, 22)}

    def sequence(clips, frames, offsets):
        return torch.zeros(len(clips), len(offsets), 109)

    algorithm.motions = SimpleNamespace(episodes=[episode], sequence=sequence)
    algorithm.robot = SimpleNamespace(
        default_q=torch.zeros(22),
        reward=lambda state, next_state, target, actions, dt: torch.ones(len(actions)),
    )
    captured = []

    def fit(states, goals, context, labels, weights, *args):
        captured.append(labels)
        return torch.tensor(0.0)

    algorithm._fit_critic = fit
    batch = SimpleNamespace(
        clips=torch.tensor([0, 0]),
        frames=torch.tensor([0, 1]),
        states=torch.zeros(2, 1, 109),
        context=torch.zeros(2, 62),
    )
    algorithm.fit_real_critic(batch)
    torch.testing.assert_close(
        captured[0], torch.tensor([1 + 0.99 + 0.99**2, 1 + 0.99])
    )


def test_changed_dataset_evaluation_seeds_require_a_new_baseline():
    from q1_soccer.train import compatible_baseline

    baseline = {"seeds": [1, 2], "verified_success_rate": 0.5}
    assert compatible_baseline(baseline, [1, 2]) is baseline
    assert compatible_baseline(baseline, [1, 3]) is None


def test_rollback_never_exits_training_when_limit_is_disabled():
    from q1_soccer.train import TrainConfig, rejection_limit_reached

    assert not rejection_limit_reached(TrainConfig(max_rejections=0), 10000)
    assert not rejection_limit_reached(TrainConfig(max_rejections=10), 9)
    assert rejection_limit_reached(TrainConfig(max_rejections=10), 10)
