"""Source warmup initialization, resume validation, and bounded training."""

from __future__ import annotations

import hashlib
import json

import pytest

from pole_task import ROOT
from pole_task.preflight import preflight
from pole_task.source_recipe import resolve_source_recipe
from pole_task.source_training import (
    SOURCE_CHECKPOINT_FORMAT,
    TrainingBudget,
    checkpoint_output_hashes,
    evaluation_cases_for,
    load_training_cases,
    select_checkpoint,
    source_acceptance,
    train_source,
    validate_resume_payload,
)

CONFIG = ROOT / "pole_task/config.json"
MANIFEST = ROOT / "pole_task/data/manifest.json"
SOURCE_MANIFEST = ROOT / "pole_task/source_manifest.json"


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(recipe, config_path=CONFIG, manifest_path=MANIFEST) -> dict:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "profile": "pole-weaving",
        "source_final_checkpoint_loaded": False,
        "source_warmup_checkpoint_sha256": recipe.warmup_checkpoint_sha256,
        "source_revision": recipe.source_revision,
        "robot_xml_sha256": manifest["robot_contract"]["xml_sha256"],
        "data_manifest_sha256": _digest(manifest_path),
        "config_sha256": _digest(config_path),
    }


def _payload(recipe, *, update=10, stage="warmup", identity=None) -> dict:
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "stage": stage,
        "update": update,
        "config": json.loads(CONFIG.read_text()),
        "recipe": recipe.to_dict(),
        "identity": _identity(recipe) if identity is None else identity,
    }


def test_preflight_strictly_loads_the_source_warmup_and_reports_provenance():
    report = preflight(CONFIG)
    recipe = resolve_source_recipe(CONFIG)

    assert report["training_ready"] is True
    assert report["training_blocker"] is None
    initialization = report["initialization"]
    assert initialization["strictly_loaded"] is True
    assert initialization["source_checkpoint_id"] == "warmup_checkpoint"
    assert initialization["sha256"] == recipe.warmup_checkpoint_sha256
    assert initialization["warmup_iteration"] == 2000
    assert initialization["source_revision"] == recipe.source_revision
    assert initialization["source_final_checkpoint_loaded"] is False
    assert initialization["final_checkpoint_sha256"] == recipe.final_checkpoint_sha256
    assert report["recipe"] == recipe.to_dict()
    assert report["recipe"]["stages"][0]["updates"] == 2000
    assert report["recipe"]["stages"][1]["updates"] == 500


def test_project_checkpoint_validation_rejects_wrong_lineage_and_stage():
    recipe = resolve_source_recipe(CONFIG)
    valid = _payload(recipe)
    assert (
        validate_resume_payload(
            valid,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )
        == 10
    )

    wrong_format = {**valid, "format": "q1_pole_weaving_project_v9"}
    with pytest.raises(ValueError, match="project-trained pole-profile"):
        validate_resume_payload(
            wrong_format,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    final_actor = _payload(
        recipe,
        identity={**_identity(recipe), "source_final_checkpoint_loaded": True},
    )
    with pytest.raises(ValueError, match="final actor"):
        validate_resume_payload(
            final_actor,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    wrong_warmup = _payload(
        recipe,
        identity={
            **_identity(recipe),
            "source_warmup_checkpoint_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="different source warmup"):
        validate_resume_payload(
            wrong_warmup,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    wrong_stage = _payload(recipe, update=2001, stage="warmup")
    with pytest.raises(ValueError, match="stage"):
        validate_resume_payload(
            wrong_stage,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    beyond_budget = _payload(recipe, update=2501, stage="actor")
    with pytest.raises(ValueError, match="Resume update"):
        validate_resume_payload(
            beyond_budget,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )


def test_checkpoint_selection_never_searches_outside_the_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        select_checkpoint(tmp_path)
    (tmp_path / "latest.pt").write_bytes(b"latest")
    assert select_checkpoint(tmp_path) == tmp_path / "latest.pt"
    (tmp_path / "best.pt").write_bytes(b"best")
    assert select_checkpoint(tmp_path) == tmp_path / "best.pt"
    assert select_checkpoint(tmp_path, prefer_best=False) == tmp_path / "latest.pt"


def test_published_checkpoints_are_hashed_for_run_provenance(tmp_path):
    (tmp_path / "latest.pt").write_bytes(b"latest")
    hashes = checkpoint_output_hashes(tmp_path)
    assert hashes == {
        "latest_checkpoint_sha256": _digest(tmp_path / "latest.pt")
    }
    (tmp_path / "best.pt").write_bytes(b"best")
    hashes = checkpoint_output_hashes(tmp_path)
    assert hashes["best_checkpoint_sha256"] == _digest(tmp_path / "best.pt")
    assert hashes["latest_checkpoint_sha256"] == _digest(tmp_path / "latest.pt")


def test_documented_preflight_is_read_only_and_opens_no_viewer(monkeypatch):
    config = json.loads(CONFIG.read_text())
    output_root = ROOT / config["output_root"]
    before = set(output_root.glob("run-*")) if output_root.exists() else set()

    def blocked(*args, **kwargs):
        raise AssertionError("preflight opened a MuJoCo viewer")

    monkeypatch.setattr("mujoco.viewer.launch_passive", blocked)
    report = preflight(CONFIG)

    assert report["training_ready"]
    assert report["initialization"]["strictly_loaded"]
    after = set(output_root.glob("run-*")) if output_root.exists() else set()
    assert after == before


def test_failed_source_preflight_creates_no_run_directory(tmp_path, monkeypatch):
    config_path = tmp_path / "pole_task/config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")

    def missing_source(_):
        raise FileNotFoundError("registered source warmup is missing")

    monkeypatch.setattr(
        "pole_task.source_training.preflight", missing_source
    )
    with pytest.raises(FileNotFoundError, match="warmup is missing"):
        train_source(config_path)
    assert not (tmp_path / "logs").exists()


def test_training_budget_is_strictly_bounded():
    budget = TrainingBudget(5)
    budget.charge(3)
    assert budget.remaining() == 2
    with pytest.raises(RuntimeError, match="environment-step budget"):
        budget.charge(3)


def test_training_cases_exclude_the_held_out_pair():
    config = json.loads(CONFIG.read_text())
    training, evaluation, all_cases = load_training_cases(config, manifest_path=MANIFEST)

    assert [case.case_id for case in evaluation] == ["eval-01"]
    assert evaluation[0].split == "held_out"
    assert all(case.split != "held_out" for case in training)
    assert {case.case_id for case in all_cases} == {
        *[case.case_id for case in training],
        "eval-01",
    }
    # The retention references are paired with their registered hashes rather
    # than being regenerated from soccer checkpoints.
    assert {case.case_id for case in training} == {
        "nominal-01",
        "straight",
        "left_turn",
        "right_turn",
    }
    assert all(case.sha256 for case in all_cases)


def _report(*, completed, collision=False, fall=False, endpoint=0.1, rmse=0.1):
    return {
        "route_completed": completed,
        "pole_collision": collision,
        "fall": fall,
        "endpoint_distance_m": endpoint,
        "root_xy_rmse_m": rmse,
    }


def test_training_evaluation_includes_every_acceptance_case():
    config = json.loads(CONFIG.read_text())
    _, _, all_cases = load_training_cases(config, manifest_path=MANIFEST)
    ordered = evaluation_cases_for(config, all_cases)
    assert [case.case_id for case in ordered] == [
        "nominal-01",
        "eval-01",
        "straight",
        "left_turn",
        "right_turn",
    ]


def test_checkpoint_selection_requires_nominal_held_out_and_retention():
    config = json.loads(CONFIG.read_text())
    good = {
        "nominal-01": _report(completed=True),
        "eval-01": _report(completed=True),
        "straight": _report(completed=True),
        "left_turn": _report(completed=True),
        "right_turn": _report(completed=True),
    }
    assert source_acceptance(good, config)
    assert not source_acceptance(
        {**good, "nominal-01": _report(completed=False)}, config
    )
    assert not source_acceptance(
        {**good, "eval-01": _report(completed=True, collision=True)}, config
    )
    assert not source_acceptance(
        {**good, "straight": _report(completed=True, fall=True)}, config
    )
    assert not source_acceptance(
        {**good, "eval-01": _report(completed=True, rmse=0.6)}, config
    )
