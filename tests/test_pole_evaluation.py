"""Fixed-seed, headless pole evaluation over the registered course set."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pole_task import COURSE_ROOT, ROOT
from pole_task.source_playback import load_course_bundle
from pole_task.source_runtime import evaluate_source_cases
from pole_task.source_training import load_training_cases

CONFIG = COURSE_ROOT / "pole_task/config.json"
MANIFEST = COURSE_ROOT / "pole_task/data/manifest.json"
EXPECTED_STEPS = {
    "nominal-01": 675,
    "eval-01": 671,
    "straight": 198,
    "left_turn": 298,
    "right_turn": 298,
}


def test_registered_references_have_the_source_step_counts_and_split():
    import json

    config = json.loads(CONFIG.read_text())
    _, evaluation, all_cases = load_training_cases(config, manifest_path=MANIFEST)

    assert {case.case_id: len(case.qpos) - 1 for case in all_cases} == EXPECTED_STEPS
    assert [case.case_id for case in evaluation] == ["eval-01"]
    by_id = {case.case_id: case for case in all_cases}
    assert by_id["nominal-01"].split == "train"
    assert by_id["eval-01"].split == "held_out"
    assert all(
        by_id[name].split == "retention"
        for name in ("straight", "left_turn", "right_turn")
    )


def test_bundle_keeps_each_registered_course_paired_with_its_motion_clip(tmp_path):
    _, motions, cases, _ = load_course_bundle(CONFIG, tmp_path / "references")
    for index, motion_case in enumerate(motions.cases):
        assert cases[motion_case.case_id].identity == motion_case.identity
        assert (motions.poles_xy[index].numpy() == motion_case.poles_xy).all()


def test_evaluation_passes_fixed_seed_zero_start_and_registered_clip(
    tmp_path, monkeypatch
):
    bundle = load_course_bundle(CONFIG, tmp_path / "references")
    _, motions, cases, _ = bundle
    case = cases["nominal-01"]
    calls = []

    def fake_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(result={"case_id": case.case_id})

    monkeypatch.setattr("pole_task.source_runtime.run_source_episode", fake_rollout)
    reports = evaluate_source_cases(
        actor=None,
        robot=None,
        motions=motions,
        cases=[case],
        scene_path=Path("/tmp/unused-scene.xml"),
        seed=4242,
    )

    assert reports == {"nominal-01": {"case_id": "nominal-01"}}
    args, kwargs = calls[0]
    assert args[4] == next(
        index for index, motion_case in enumerate(motions.cases)
        if motion_case.case_id == "nominal-01"
    )
    assert kwargs["start_frame"] == 0
    assert kwargs["seed"] == 4242
    assert kwargs["action_noise"] == 0.0
    assert kwargs["observation_noise"] is False
