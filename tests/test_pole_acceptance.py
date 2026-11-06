"""Twenty fixed-seed nominal trials and combined pole acceptance."""

from __future__ import annotations

from types import SimpleNamespace

from pole_task.acceptance import nominal_trial_passes, run_nominal_trials

CONFIG = {
    "course_success_endpoint_m": 0.5,
    "course_success_rmse_m": 0.5,
}


def _row(**overrides):
    row = {
        "route_completed": True,
        "pole_collision": False,
        "fall": False,
        "endpoint_distance_m": 0.1,
        "root_xy_rmse_m": 0.1,
    }
    row.update(overrides)
    return row


def test_nominal_trial_acceptance_requires_completion_clearance_and_tracking():
    assert nominal_trial_passes(_row(), CONFIG)
    assert not nominal_trial_passes(_row(route_completed=False), CONFIG)
    assert not nominal_trial_passes(_row(pole_collision=True), CONFIG)
    assert not nominal_trial_passes(_row(fall=True), CONFIG)
    assert not nominal_trial_passes(_row(endpoint_distance_m=0.51), CONFIG)
    assert not nominal_trial_passes(_row(root_xy_rmse_m=0.51), CONFIG)


def test_twenty_nominal_trials_use_distinct_fixed_seeds_and_report_failures(monkeypatch):
    case = SimpleNamespace(case_id="nominal-01", identity="nominal-01:abc")
    motions = SimpleNamespace(cases=(case,))
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(result=_row(fall=len(calls) == 6))

    monkeypatch.setattr("pole_task.acceptance.run_source_episode", fake_rollout)
    rows = run_nominal_trials(
        actor=None,
        robot=None,
        motions=motions,
        case=case,
        scene_path="unused.xml",
        config=CONFIG,
        trials=20,
        seed=700,
    )

    assert len(rows) == 20
    assert [row["seed"] for row in rows] == list(range(700, 720))
    assert all(row["passed"] for index, row in enumerate(rows) if index != 5)
    assert not rows[5]["passed"]
    assert all(call["start_frame"] == 0 for call in calls)
    assert all(call["action_noise"] == 0.0 for call in calls)
    assert all(call["observation_noise"] is False for call in calls)
