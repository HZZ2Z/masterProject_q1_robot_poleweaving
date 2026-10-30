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
            terminal, terminal)


def toy_agent(sign=1.):
    agent = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    agent.dynamics = torch.nn.Linear(34, 12)
    with torch.no_grad():
        agent.dynamics.weight.zero_(); agent.dynamics.bias.zero_()
        agent.dynamics.weight[0, 12] = sign
        for p in agent.critic.parameters():p.zero_()
    agent.task_reward = lambda c, n, a: (n[:, 0], n.new_ones(len(n)))
    agent.make_optimizers()
    return agent


def test_model_actor_uses_dynamics_and_preserves_model_parameters():
    torch.manual_seed(7)
    a, b = toy_agent(1), toy_agent(-1)
    states = torch.zeros(8, 12)
    model = copy.deepcopy(a.dynamics.state_dict())
    a.improve_model_policy(states, 2, .05)
    b.improve_model_policy(states, 2, .05)
    assert a.act(states)[0, 0] > 0 and b.act(states)[0, 0] < 0
    for k in model:torch.testing.assert_close(a.dynamics.state_dict()[k], model[k])


def test_learned_reward_head_cannot_manipulate_actor_objective():
    torch.manual_seed(8)
    a = toy_agent(); b = copy.deepcopy(a); b.make_optimizers()
    with torch.no_grad():
        for p in b.outcomes.parameters():p.mul_(100).add_(100)
    states = torch.zeros(8, 12)
    torch.manual_seed(9); a.improve_model_policy(states, 2, .05)
    torch.manual_seed(9); b.improve_model_policy(states, 2, .05)
    for p, q in zip(a.actor.parameters(), b.actor.parameters()):torch.testing.assert_close(p, q)


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


def test_model_policy_respects_kl_and_full_action_range():
    a = toy_agent()
    a.improve_model_policy(torch.zeros(16, 12), 2, .05, max_kl=1e-5)
    assert a.last_diagnostics['learning/policy_kl'] <= 1e-5
    with torch.no_grad():a.actor[-1].bias.fill_(2.)
    assert a.act(torch.zeros(1, 12)).min() > .9


def test_value_fit_reduces_fixed_target_error():
    torch.manual_seed(3)
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2, lr=.001)
    a.make_optimizers()
    states = torch.ones(16, 12)
    batch = list(rollout(states)); batch[3] = torch.ones(1, 16)
    a.fit_real_values(tuple(batch), epochs=8)
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


def test_policy_rejection_restores_actor_but_retains_value_learning(monkeypatch):
    from q1_supermimic import safety
    baseline = {'duration_s': 6., 'return': 5., 'survival_5s': 1., 'fall_fraction': 0.}
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: dict(baseline))
    a = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    a.make_optimizers()
    a.fit_real_values(rollout(torch.zeros(8, 12)), epochs=1)
    gate = SafetyGate(a, count=4, steps=10)
    a.fit_real_values(rollout(torch.zeros(8, 12), -1), epochs=1)
    optimizer = copy.deepcopy(a.critic_optimizer.state_dict())
    actor = copy.deepcopy(gate.accepted_actor); critic = copy.deepcopy(a.critic.state_dict())
    with torch.no_grad():
        for p in a.actor.parameters():p.add_(1.)
        a.action_limit.fill_(.8)
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: {**baseline, 'duration_s': .2, 'return': -20., 'survival_5s': 0., 'fall_fraction': 1.})
    result = gate.check(a)
    assert result['evaluation/policy_accepted'] == 0
    assert abs(float(a.action_limit)-1.) < 1e-6
    for k in actor:torch.testing.assert_close(a.actor.state_dict()[k], actor[k])
    for k in critic:torch.testing.assert_close(a.critic.state_dict()[k], critic[k])

    for key, value in optimizer['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(a.critic_optimizer.state_dict()['state'][key][name], tensor)


def test_balance_survival_tolerance_accounts_for_finite_cohort():
    from q1_supermimic.safety import survival_floor
    floor = survival_floor(.5, 32)
    assert .28 < floor < .46875
    assert survival_floor(1., 32) > .85
    assert survival_floor(.5, 128) > floor


def test_model_updates_bound_cumulative_kl_since_real_acceptance():
    import copy
    agent = toy_agent(1.)
    initial = torch.zeros(32, 12)
    reference = copy.deepcopy(agent.actor.state_dict())
    obs = agent.policy_observation(agent.normalize(initial))
    with torch.no_grad(): before = agent.actor(obs).clone()
    for _ in range(12):
        agent.improve_model_policy(initial, 2, .025, max_kl=.001, reference_state=reference)
    with torch.no_grad(): kl = ((agent.actor(obs)-before).square()/(2*.025**2)).sum(-1).mean()
    assert kl <= .00101


def test_rejected_step_reports_existing_reference_distance():
    import copy
    agent = toy_agent(1.)
    reference = copy.deepcopy(agent.actor.state_dict())
    with torch.no_grad(): agent.actor[-1].bias.fill_(.1)
    before = copy.deepcopy(agent.actor.state_dict())
    agent.improve_model_policy(torch.zeros(8, 12), 2, .025, max_kl=.001, reference_state=reference)
    assert agent.last_diagnostics['learning/update_fraction'] == 0
    assert agent.last_diagnostics['learning/policy_kl'] > .001
    for k,v in agent.actor.state_dict().items(): torch.testing.assert_close(v,before[k])


def test_stall_watchdog_checks_applied_and_retained_updates():
    from train import policy_stall_reason
    assert policy_stall_reason(249, 49, 49, 200).startswith('No actor optimizer')
    assert policy_stall_reason(249, 240, 49, 200).startswith('No changed actor')
    assert policy_stall_reason(249, 240, 200, 200) is None
    assert policy_stall_reason(1, 49, 49, 200) is None


def test_unchanged_actor_is_not_reported_as_new_acceptance(monkeypatch):
    import q1_supermimic.safety as safety
    agent = toy_agent()
    metrics = {'duration_s':6., 'return':1., 'survival_5s':1., 'fall_fraction':0.}
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *args, **kwargs: dict(metrics))
    gate = safety.SafetyGate(agent)
    result = gate.check(agent)
    assert result['evaluation/policy_accepted'] == 0
    assert result['evaluation/rollback_count'] == 0
    with torch.no_grad(): agent.actor[-1].bias.add_(.001)
    assert gate.check(agent)['evaluation/policy_accepted'] == 1


def test_gate_compares_incumbent_on_identical_rotating_cohorts(monkeypatch):
    import q1_supermimic.safety as safety
    agent = toy_agent()
    calls = []
    def measure(agent, progress, count, steps, seed, **kwargs):
        calls.append((seed, kwargs.get('noise_std', 0.), float(agent.actor[-1].bias[0].detach())))
        return {'return': 1., 'survival_5s': 1., 'fall_fraction': 0.}
    monkeypatch.setattr(safety, 'evaluate_actor', measure)
    gate = safety.SafetyGate(agent, seeds=[11, 23], noise_std=.025)
    calls.clear()
    with torch.no_grad(): agent.actor[-1].bias.add_(.01)
    assert gate.check(agent)['evaluation/policy_accepted'] == 1
    assert [(s, n) for s,n,_ in calls[:4]] == [(s,n) for s,n,_ in calls[4:]]
    assert all(b == 0. for _,_,b in calls[:4])
    assert all(b > 0. for _,_,b in calls[4:])
    previous_seeds = {s for s,_,_ in calls}
    calls.clear()
    gate.check(agent)
    assert not previous_seeds.intersection(s for s,_,_ in calls)
