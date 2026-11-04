import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pole_task import ROOT
from pole_task.source import verify_source_manifest
from pole_task.source_checkpoint import (
    SOURCE_ACTOR_PARITY_TOLERANCE,
    load_source_pole,
)
from pole_task.source_profile import SOURCE_ACTION_DIM, source_goal_features

MANIFEST_PATH = ROOT / "pole_task/source_manifest.json"


def _fixture() -> dict[str, np.ndarray]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    path = ROOT / "pole_task" / manifest["policy_fixture"]["project_path"]
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_loader_reproduces_source_fixture_within_registered_tolerance():
    fixture = _fixture()
    loaded = load_source_pole(MANIFEST_PATH)
    robot = loaded.robot
    robot.set_course_frame((0.0, 0.0, 0.0), 0.0, 0.04)
    state = torch.from_numpy(fixture["state"]).unsqueeze(0)
    context = torch.from_numpy(fixture["context"]).unsqueeze(0)
    references = torch.from_numpy(fixture["references"]).unsqueeze(0)
    history = torch.from_numpy(fixture["history"]).unsqueeze(0)

    observation = robot.observation(state, context)
    goal = source_goal_features(robot, references, state)
    with torch.no_grad():
        action = loaded.actor(history, goal)

    tolerance = SOURCE_ACTOR_PARITY_TOLERANCE
    assert tolerance == loaded.provenance["actor_parity_tolerance"]
    assert (observation[0] - torch.from_numpy(fixture["observation"])).abs().max() <= tolerance
    assert (goal[0] - torch.from_numpy(fixture["goal"])).abs().max() <= tolerance
    assert (action[0] - torch.from_numpy(fixture["action"])).abs().max() <= tolerance
    assert action.shape == (1, SOURCE_ACTION_DIM)


def test_strict_loader_records_deterministic_mappings_and_source_identity():
    loaded = load_source_pole(MANIFEST_PATH)
    provenance = loaded.provenance

    assert provenance["source_revision"] == "e023c3147e5c20af17f329e0b467538f627a27cb"
    assert provenance["source_task"] == "7.7"
    assert provenance["warmup_checkpoint_sha256"] == (
        "b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5"
    )
    assert provenance["final_checkpoint_reference_sha256"] == (
        "c42237dff52a5664d83af37fa7afed00670d09955f562855161dc59f7b73f8d6"
    )
    assert provenance["source_final_checkpoint_loaded"] is False
    assert provenance["robot_xml_sha256"] == (
        "d9442f0de18071b100d927e310bc58b04e7be305fea8f126893b2700e8c4bd3b"
    )
    assert provenance["history_order"] == "oldest_to_newest"
    assert provenance["state_dict_mapping"].startswith("identity")
    assert provenance["excluded_checkpoint_absent_modules"] == [
        "speed_encoder",
        "route_encoder",
        "route_adapter",
    ]
    assert provenance["warmup_iteration"] == 2000
    assert "warmup_checkpoint" in provenance["verified_artifacts"]


def test_strict_loader_rejects_the_source_final_actor_as_initialization(tmp_path):
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["source_root"] = str(
        (MANIFEST_PATH.parent / manifest["source_root"]).resolve()
    )
    manifest["warmup_checkpoint_id"] = manifest["final_checkpoint_reference_id"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="final actor must not be used"):
        load_source_pole(path)


def _minimal_source_manifest(tmp_path: Path, checkpoint: dict) -> Path:
    real = json.loads(MANIFEST_PATH.read_text())
    source = tmp_path / "source"
    (source / "assets").mkdir(parents=True)
    real_report = verify_source_manifest(MANIFEST_PATH)
    robot_xml = source / "q1_v3_mc.xml"
    robot_xml.write_bytes(Path(real_report["artifact_paths"]["robot_xml"]).read_bytes())
    warmup_path = source / "warmup.pt"
    torch.save(checkpoint, warmup_path)
    final_path = source / "final.pt"
    final_path.write_bytes(b"final reference is never loaded")
    config_path = source / "warmup.toml"
    config_path.write_text("[train]\n")

    manifest = dict(real)
    manifest["source_root"] = str(source)
    manifest["robot_contract"] = dict(real["robot_contract"])
    manifest["robot_contract"]["xml_sha256"] = _sha256(robot_xml)
    manifest["training_contract"] = dict(real["training_contract"])
    manifest["training_contract"]["warmup_config_sha256"] = _sha256(config_path)
    manifest["artifacts"] = [
        {
            "id": real["warmup_checkpoint_id"],
            "path": "warmup.pt",
            "sha256": _sha256(warmup_path),
        },
        {
            "id": real["final_checkpoint_reference_id"],
            "path": "final.pt",
            "sha256": _sha256(final_path),
        },
        {"id": "robot_xml", "path": "q1_v3_mc.xml", "sha256": _sha256(robot_xml)},
        {"id": "warmup_config", "path": "warmup.toml", "sha256": _sha256(config_path)},
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_strict_loader_rejects_incompatible_actor_weight_shapes(tmp_path):
    real_report = verify_source_manifest(MANIFEST_PATH)
    checkpoint = torch.load(
        Path(real_report["artifact_paths"][real_report["warmup_checkpoint_id"]]),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["actor"]["encoder.0.weight"] = torch.zeros(7, 7)
    manifest_path = _minimal_source_manifest(tmp_path, checkpoint)

    with pytest.raises(ValueError, match="actor state dictionary differs"):
        load_source_pole(manifest_path)


def test_strict_loader_rejects_a_checkpoint_with_a_different_contract(tmp_path):
    real_report = verify_source_manifest(MANIFEST_PATH)
    checkpoint = torch.load(
        Path(real_report["artifact_paths"][real_report["warmup_checkpoint_id"]]),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["config"]["goal_representation"] = "absolute"
    manifest_path = _minimal_source_manifest(tmp_path, checkpoint)

    with pytest.raises(ValueError, match="goal_representation"):
        load_source_pole(manifest_path)
