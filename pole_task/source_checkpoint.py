"""Strict loading of the registered source task 7.7 warmup checkpoint.

The loader verifies the migration manifest and every registered artifact hash
before reading weights.  It then checks the checkpoint's recorded model
contract, builds the source-compatible robot and networks, and loads every
state dictionary with ``strict=True``.  No key is renamed, projected or
silently skipped, and the source task's final actor is never accepted as an
initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .source import verify_source_manifest
from .source_profile import SourcePoleRobot, build_source_models

SOURCE_ACTOR_PARITY_TOLERANCE = 1e-6

_EXPECTED_STATE_DIM = 79
_EXPECTED_OBS_DIM = 78
_EXPECTED_CONTEXT_DIM = 149
_EXPECTED_ACTION_SCALE = 0.5
_EXPECTED_ARCHITECTURE = {
    "activation": "silu",
    "representation": "heading",
    "modeled_action_delay": True,
}


@dataclass(frozen=True)
class LoadedSourcePole:
    """Strictly loaded source profile components and their provenance."""

    robot: SourcePoleRobot
    actor: nn.Module
    world_model: nn.Module
    critic: nn.Module
    checkpoint: dict[str, Any]
    provenance: dict[str, Any]


def _validate_state_dict(
    name: str, state: dict[str, torch.Tensor], expected: dict[str, list[int]]
) -> None:
    actual = {key: list(value.shape) for key, value in state.items()}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        raise ValueError(
            f"Source {name} state dictionary differs from the registered contract: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )


def _validate_robot_state_dict(state: dict[str, Any], expected: dict[str, list[int]]) -> None:
    unexpected = set(state) - set(expected) - {"_extra_state"}
    missing = set(expected) - set(state)
    if unexpected or missing:
        raise ValueError(
            "Source robot state dictionary differs from the registered contract: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, shape in expected.items():
        value = state[key]
        if list(value.shape) != shape:
            raise ValueError(
                f"Source robot tensor {key} has shape {list(value.shape)}, expected {shape}"
            )
    if "_extra_state" not in state:
        raise ValueError("Source robot checkpoint is missing reward-scaling metadata")


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    final_sha256: str,
    warmup_sha256: str,
) -> None:
    for key in ("config", "identity", "actor", "model", "critic", "robot", "iteration"):
        if key not in checkpoint:
            raise ValueError(f"Source warmup checkpoint is missing {key!r}")
    if warmup_sha256 == final_sha256:
        raise ValueError("Registered warmup and final checkpoint identities must differ")
    config = checkpoint["config"]
    identity = checkpoint["identity"]
    expected_config = {
        "action_scale": _EXPECTED_ACTION_SCALE,
        "modeled_action_delay": True,
        "goal_representation": "robot",
        "q1_reward_scaling": True,
        "network_activation": "silu",
        "dynamics_representation": "heading",
        "normalize_policy_objective": True,
        "normalize_model_objective": True,
        "small_networks": False,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(
                f"Source warmup config {key!r} is {config.get(key)!r}, expected {value!r}"
            )
    for key, value in (
        ("state_dim", _EXPECTED_STATE_DIM),
        ("obs_dim", _EXPECTED_OBS_DIM),
        ("context_dim", _EXPECTED_CONTEXT_DIM),
    ):
        if identity.get(key) != value:
            raise ValueError(
                f"Source warmup identity {key!r} is {identity.get(key)!r}, expected {value!r}"
            )
    if identity.get("network_architecture") != _EXPECTED_ARCHITECTURE:
        raise ValueError(
            "Source warmup identity network_architecture differs from the "
            "registered activation/representation/delay contract"
        )
    if identity.get("xml_sha256") != manifest["robot_contract"]["xml_sha256"]:
        raise ValueError("Source warmup robot XML identity differs from the migration manifest")
    if identity.get("control_dt") != 0.02:
        raise ValueError("Source warmup control timestep is not 0.02 s")
    tuning = identity.get("reward_tuning", {})
    if tuning.get("action_scale") != _EXPECTED_ACTION_SCALE:
        raise ValueError("Source warmup reward tuning action scale differs from 0.5")
    if tuning.get("pole_clearance_weight") != 0.0:
        raise ValueError("Source warmup uses an unexpected pole clearance reward")
    if config.get("reward_pole_clearance_weight") != 0.0:
        raise ValueError("Source warmup config enables a pole clearance reward")


def load_source_pole(
    manifest_path: Path,
    *,
    device: str = "cpu",
) -> LoadedSourcePole:
    """Verify the manifest and strictly load the registered warmup checkpoint."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    report = verify_source_manifest(manifest_path)
    artifact_paths = report["artifact_paths"]
    warmup_id = manifest["warmup_checkpoint_id"]
    final_id = manifest["final_checkpoint_reference_id"]
    warmup_sha = next(
        artifact["sha256"] for artifact in manifest["artifacts"] if artifact["id"] == warmup_id
    )
    final_sha = next(
        artifact["sha256"] for artifact in manifest["artifacts"] if artifact["id"] == final_id
    )
    warmup_path = Path(artifact_paths[warmup_id]).resolve()
    if Path(artifact_paths[final_id]).resolve() == warmup_path:
        raise ValueError("Source final actor must not be used as training initialization")

    checkpoint = torch.load(warmup_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Source warmup checkpoint must be a mapping")
    _validate_checkpoint_metadata(checkpoint, manifest, final_sha, warmup_sha)

    shapes = manifest["tensor_shapes"]
    _validate_state_dict("actor", checkpoint["actor"], shapes["actor"])
    _validate_state_dict("model", checkpoint["model"], shapes["model"])
    _validate_state_dict("critic", checkpoint["critic"], shapes["critic"])
    _validate_robot_state_dict(checkpoint["robot"], shapes["robot"])

    robot_xml = Path(artifact_paths["robot_xml"]).resolve()
    robot = SourcePoleRobot(robot_xml)
    if robot.sha256 != manifest["robot_contract"]["xml_sha256"]:
        raise ValueError("Source robot XML hash differs from the migration manifest")
    robot.load_state_dict(checkpoint["robot"], strict=True)
    if checkpoint["identity"].get("reward_scaling") != robot.reward_identity():
        raise ValueError("Source warmup reward scaling differs from the Q1 XML identity")
    actor, world_model, critic = build_source_models(robot)
    expected_obs_scale = torch.tensor(
        [1.0] * 22 + [10.0] * 22 + [3.0] * 3 + [1.0] * 25 + [1.0] * 6
    )
    if not torch.equal(actor.obs_scale, expected_obs_scale):
        raise ValueError("Source actor observation scale differs from the normalization contract")
    actor.load_state_dict(checkpoint["actor"], strict=True)
    world_model.load_state_dict(checkpoint["model"], strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)

    actor.to(device)
    world_model.to(device)
    critic.to(device)
    robot.to(device)
    provenance = {
        "source_revision": manifest["source_revision"],
        "source_change": manifest["source_change"],
        "source_task": manifest["source_task"],
        "warmup_checkpoint_id": warmup_id,
        "warmup_checkpoint_path": str(warmup_path),
        "warmup_checkpoint_sha256": warmup_sha,
        "warmup_iteration": int(checkpoint["iteration"]),
        "final_checkpoint_reference_id": final_id,
        "final_checkpoint_reference_sha256": final_sha,
        "source_final_checkpoint_loaded": False,
        "source_config": artifact_paths.get("warmup_config"),
        "source_config_sha256": manifest["training_contract"]["warmup_config_sha256"],
        "robot_xml": str(robot_xml),
        "robot_xml_sha256": manifest["robot_contract"]["xml_sha256"],
        "state_dict_mapping": "identity for actor, world model, critic and robot; no renamed or skipped keys",
        "observation_mapping": (
            f"{_EXPECTED_OBS_DIM} values: 22 joints + 22 joint velocities + 3 gyro + "
            "3 gravity + 22 previous actions + 6 heading-frame pole offsets"
        ),
        "normalization_mapping": (
            "actor obs_scale is 78 values (72 vendored values plus six 1.0 pole "
            "columns); world model/critic state scale is 79 values "
            "(1 x 29, 3 x 6, 10 x 22, 1 x 22); world model context scale is "
            "ones(149) with the delay column scaled by 0.016"
        ),
        "goal_mapping": "10 future reference frames x 57 physical values in the robot root frame",
        "context_mapping": "149 columns; course pole XY centers occupy columns 143:149",
        "history_order": "oldest_to_newest",
        "excluded_checkpoint_absent_modules": [
            "speed_encoder",
            "route_encoder",
            "route_adapter",
        ],
        "actor_parity_tolerance": SOURCE_ACTOR_PARITY_TOLERANCE,
        "verified_artifacts": report["verified_artifacts"],
    }
    return LoadedSourcePole(
        robot=robot,
        actor=actor,
        world_model=world_model,
        critic=critic,
        checkpoint=checkpoint,
        provenance=provenance,
    )


__all__ = ["LoadedSourcePole", "SOURCE_ACTOR_PARITY_TOLERANCE", "load_source_pole"]
