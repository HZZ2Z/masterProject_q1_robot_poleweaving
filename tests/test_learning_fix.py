"""Regressions for the failed learned-reward imagination update."""
import copy
import torch
from q1_supermimic.learning import SoccerAgent, TemporalReplay
from q1_supermimic.safety import SafetyGate


def test_actor_does_not_optimize_learned_reward_or_termination_head():
    torch.manual_seed(7)
    first = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    second = copy.deepcopy(first)
    with torch.no_grad():
        for p in second.outcomes.parameters():
            p.mul_(100).add_(100)
    first.make_optimizers()
    second.make_optimizers()
    state = torch.randn(8, 12)
    first.improve(state)
    second.improve(state)
    for a, b in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (first.act(state).abs() <= .1).all()


def test_real_critic_targets_follow_actual_terminal_reward():
    torch.manual_seed(8)
    positive = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    with torch.no_grad():
        positive.critic[-1].weight.zero_()
        positive.critic[-1].bias.zero_()
    negative = copy.deepcopy(positive)
    positive.make_optimizers()
    negative.make_optimizers()
    states = torch.randn(8, 2, 12)
    actions = torch.zeros(8, 2, 22)
    terminal = torch.ones(8, 2)
    positive.fit_critic((states, actions, states, terminal * 10, terminal, terminal))
    negative.fit_critic((states, actions, states, terminal * -10, terminal, terminal))
    assert positive.critic[-1].bias.item() > 0
    assert negative.critic[-1].bias.item() < 0


def test_validation_worlds_never_enter_model_training_batches():
    replay = TemporalReplay(4, 16, 3, 'cpu')
    for _ in range(4):
        state = torch.arange(16).float()[:, None].expand(16, 3)
        replay.add(state, torch.zeros(16, 22), state, torch.zeros(16), torch.zeros(16), torch.zeros(16))
    train = replay.sample(500, 2)[0]
    held_out = replay.sample(500, 2, validation=True)[0]
    assert train.max() <= 13 and held_out.min() >= 14


def test_real_regression_rolls_back_actor_and_optimizer(monkeypatch):
    from q1_supermimic import safety
    baseline = {'duration_s': 6., 'return': 5., 'survival_5s': 1., 'fall_fraction': 0.}
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: dict(baseline))
    agent = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    agent.make_optimizers()
    gate = SafetyGate(agent, count=4, steps=10)
    before = copy.deepcopy(agent.actor.state_dict())
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(1.)
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: {**baseline, 'duration_s': .2, 'return': -20.})
    measured = gate.check(agent)
    assert measured['evaluation/policy_accepted'] == 0
    assert measured['evaluation/rollback_count'] == 1
    for name, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_real_policy_gradient_responds_to_actual_action_reward_correlation():
    torch.manual_seed(19)
    first = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    with torch.no_grad():
        first.critic[-1].weight.zero_()
        first.critic[-1].bias.zero_()
    second = copy.deepcopy(first)
    first.make_optimizers()
    second.make_optimizers()
    states = torch.zeros(16, 12)
    sign = torch.tensor([1., -1.] * 8)
    actions = torch.zeros(16, 22)
    actions[:, 0] = .03 * sign
    terminal = torch.ones(16)
    first.improve(states, (states, actions, states, sign, terminal, terminal))
    second.improve(states, (states, actions, states, -sign, terminal, terminal))
    assert first.act(states)[0, 0] > 0
    assert second.act(states)[0, 0] < 0
