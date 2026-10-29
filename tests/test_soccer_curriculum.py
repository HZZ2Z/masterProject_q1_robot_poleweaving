import copy
import torch
from q1_supermimic.environment import load_environment
from q1_supermimic.learning import SoccerAgent
from q1_supermimic.safety import SafetyGate


def test_cohort_gate_ignores_synchronized_young_episodes_and_caps_practice():
    e = load_environment('cpu', 8).env
    e.episode_length_buf.zero_()
    e.survival_rate_5s = .72
    e.set_training_iteration(5000)
    assert not e.balance_phase
    e._curriculum_quality_window = []
    e._curriculum_last_adjustment = None
    for i in range(5001, 5501):
        e.set_training_iteration(i)
    assert abs(e.soccer_progress - .35) < 1e-6
    assert e.full_soccer_progress == 0
    assert (e.task_progress[e.reset_group == 2] == 1).all()
    assert (e.task_progress[e.reset_group != 2] == 0).all()


def test_idle_near_practice_has_no_standing_income_and_timeout_is_not_fall():
    e = load_environment('cpu', 8).env
    e.balance_release_iteration = 0
    e.soccer_progress = .1
    e.set_training_iteration(5000)
    near = e.reset_group == 2
    e.near_ball_stall[:] = 1
    e.far_ball_stall[:] = 1
    r = e._reward_survival()
    assert (r[near] == 0).all() and (r[~near] == 1).all()
    e.episode_length_buf[:] = 601
    e._check_termination()
    assert e.time_out_buf[near].all() and e.reset_buf[near].all()
    assert not e.extras['fall'][near].any()
    assert not e.time_out_buf[~near].any()


def test_soccer_gate_prefers_task_progress_to_standing_return(monkeypatch):
    from q1_supermimic import safety
    standing = dict(duration_s=6., return_=20., survival_5s=1., fall_fraction=0.,
                    near_approach_m=0., near_kick_event_rate=0., near_net_goal_progress_m=0.,
                    near_success_rate=0., near_sample_count=2, task_score=0.)
    standing['return'] = standing.pop('return_')
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *a, **kw: dict(standing))
    agent = SoccerAgent(12, obs_dim=5, width=16, horizon=2)
    agent.make_optimizers()
    gate = SafetyGate(agent, count=4)
    gate.check(agent, .1)
    assert gate.soccer_best_actor is None
    moving = dict(standing, task_score=.2, near_approach_m=.1,
                  near_kick_event_rate=.25, near_net_goal_progress_m=.02)
    moving['return'] = -1.
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *a, **kw: dict(moving))
    result = gate.check(agent, .2)
    assert result['evaluation/policy_accepted'] == 1
    assert gate.soccer_best_actor is not None
    saved = copy.deepcopy(gate.soccer_best_actor)
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *a, **kw: dict(standing))
    result = gate.check(agent, .3)
    # Safe task-score fluctuations no longer reset the learning policy.
    assert result['evaluation/policy_accepted'] == 1
    for k in saved:
        torch.testing.assert_close(saved[k], gate.soccer_best_actor[k])
    falling = dict(moving, task_score=10., survival_5s=.2, fall_fraction=.8)
    monkeypatch.setattr(safety, 'evaluate_actor', lambda *a, **kw: dict(falling))
    assert gate.check(agent, .3)['evaluation/policy_accepted'] == 0
