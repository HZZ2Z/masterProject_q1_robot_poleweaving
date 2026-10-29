"""Regressions for credit assignment, policy consistency and safe rollback."""
import copy
import torch
from q1_supermimic.learning import SoccerAgent, TemporalReplay, real_gae
from q1_supermimic.safety import SafetyGate


def rollout(states, sign=1.):
    latent = torch.zeros(1, len(states), 22)
    latent[0, :, 0] = torch.tensor([.3, -.3] * (len(states)//2))
    terminal = torch.ones(1, len(states))
    return (states[None], .1*latent.tanh(), states[None], sign*latent[..., 0],
            terminal, terminal, latent)


def test_actor_is_independent_of_learned_model_gradients():
    torch.manual_seed(7)
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    b = copy.deepcopy(a)
    with torch.no_grad():
        for p in list(b.outcomes.parameters()) + list(b.dynamics.parameters()):
            p.mul_(100).add_(100)
    a.make_optimizers(); b.make_optimizers()
    batch = rollout(torch.randn(8, 12))
    torch.manual_seed(8); a.learn_real_rollout(batch, .01, critic_epochs=1)
    torch.manual_seed(8); b.learn_real_rollout(batch, .01, critic_epochs=1)
    for p, q in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_credit_assignment_reaches_earlier_actions_and_stops_at_reset():
    rewards = torch.tensor([[0.], [0.], [1.]])
    zero = torch.zeros_like(rewards)
    done = torch.tensor([[0.], [1.], [1.]])
    adv, ret = real_gae(rewards, zero, zero, done, done, gamma=1, lam=1)
    torch.testing.assert_close(adv, rewards)  # next episode reward cannot leak
    done[1] = 0
    adv, _ = real_gae(rewards, zero, zero, done, done, gamma=1, lam=1)
    torch.testing.assert_close(adv, torch.ones_like(rewards))
    assert not ret.requires_grad


def test_timeout_bootstraps_terminal_observation_but_success_does_not():
    reward = torch.tensor([[1., 1., 1.]])
    v = torch.zeros_like(reward)
    nv = torch.full_like(reward, 10.)
    done = torch.ones_like(reward)
    terminated = torch.tensor([[0., 1., 1.]])  # timeout, success, fall
    _, ret = real_gae(reward, v, nv, done, terminated, gamma=.9, lam=.95)
    torch.testing.assert_close(ret, torch.tensor([[10., 1., 1.]]))


def test_policy_gradient_uses_real_action_reward_and_respects_kl():
    torch.manual_seed(19)
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    b = copy.deepcopy(a)
    a.make_optimizers(); b.make_optimizers()
    state = torch.zeros(16, 12)
    a.learn_real_rollout(rollout(state, 1), .01, critic_epochs=1, max_kl=1e-5)
    b.learn_real_rollout(rollout(state, -1), .01, critic_epochs=1, max_kl=1e-5)
    assert a.act(state)[0, 0] > 0 and b.act(state)[0, 0] < 0
    assert a.last_diagnostics['learning/policy_kl'] <= 1e-5
    assert b.last_diagnostics['learning/policy_kl'] <= 1e-5


def test_value_fit_reduces_fixed_target_error():
    torch.manual_seed(3)
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2, lr=.001)
    a.make_optimizers()
    states = torch.ones(16, 12)
    batch = list(rollout(states)); batch[3] = torch.ones(1, 16)
    a.learn_real_rollout(tuple(batch), .01, critic_epochs=8, update_actor=False)
    d = a.last_diagnostics
    assert d['learning/value_mse_after'] < d['learning/value_mse_before']


def test_validation_worlds_never_enter_model_training_batches():
    replay = TemporalReplay(4, 16, 3, 'cpu')
    for _ in range(4):
        state = torch.arange(16).float()[:, None].expand(16, 3)
        replay.add(state, torch.zeros(16, 22), state, torch.zeros(16), torch.zeros(16), torch.zeros(16))
    train = replay.sample(500, 2)[0]
    held_out = replay.sample(500, 2, validation=True)[0]
    assert train.max() <= 13 and held_out.min() >= 14


def test_real_regression_restores_complete_policy_and_critic(monkeypatch):
    from q1_supermimic import safety
    baseline = {'duration_s': 6., 'return': 5., 'survival_5s': 1., 'fall_fraction': 0.}
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: dict(baseline))
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    a.make_optimizers()
    a.learn_real_rollout(rollout(torch.zeros(8, 12)), .01, critic_epochs=1)
    gate = SafetyGate(a, count=4, steps=10)
    optimizer = copy.deepcopy(a.critic_optimizer.state_dict())
    a.learn_real_rollout(rollout(torch.zeros(8, 12), -1), .01, critic_epochs=1)
    actor = copy.deepcopy(gate.accepted_actor); critic = copy.deepcopy(gate.accepted_critic)
    with torch.no_grad():
        for p in list(a.actor.parameters())+list(a.critic.parameters()):p.add_(1.)
        a.action_limit.fill_(.8)
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: {**baseline, 'duration_s': .2, 'return': -20.})
    result = gate.check(a)
    assert result['evaluation/policy_accepted'] == 0
    assert abs(float(a.action_limit)-.1) < 1e-6
    for k in actor:torch.testing.assert_close(a.actor.state_dict()[k], actor[k])
    for k in critic:torch.testing.assert_close(a.critic.state_dict()[k], critic[k])

    for key, value in optimizer['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(a.critic_optimizer.state_dict()['state'][key][name], tensor)
