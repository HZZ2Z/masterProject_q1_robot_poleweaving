"""Strict loading of project-trained source-profile pole checkpoints.

Playback and evaluation never search for a soccer checkpoint, the legacy expert
checkpoint, or the source task 7.7 final actor.  A checkpoint is accepted only
when it carries the pole profile's format, source-warmup lineage, robot/data
identity, and a strict source actor state dictionary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from . import ROOT
from .source import source_robot_xml, verify_source_manifest
from .source_profile import SourcePoleRobot
from .source_recipe import resolve_source_recipe
from .source_training import SOURCE_CHECKPOINT_FORMAT
from .models import build_source_models

SOURCE_ACTOR_PARITY_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pole_actor(
    checkpoint_path: Path,
    device: str = "cpu",
    *,
    config_path: Path = ROOT / "pole_task/config.json",
) -> tuple[torch.nn.Module, dict]:
    """Load a source-profile actor from a project-trained pole checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pole checkpoint not found: {checkpoint_path}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("Pole checkpoint must be a mapping")
    checkpoint_format = saved.get("format")
    if checkpoint_format == "primebot_supermimic_soccer_v1":
        raise ValueError(
            "Soccer checkpoint cannot drive pole playback; a project-trained "
            "pole-profile checkpoint is required"
        )
    if checkpoint_format is None and {"config", "identity", "iteration"} <= set(saved):
        raise ValueError(
            "Source task 7.7 checkpoint is a reference artifact, never a pole "
            "playback fallback; train the project pole profile first"
        )
    if checkpoint_format != SOURCE_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format {checkpoint_format!r} is not a project-trained "
            f"pole policy ({SOURCE_CHECKPOINT_FORMAT!r})"
        )
    identity = saved.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Pole checkpoint is missing its identity")
    if identity.get("profile") != "pole-weaving":
        raise ValueError(
            f"Pole checkpoint profile {identity.get('profile')!r} is not pole-weaving"
        )
    if identity.get("source_final_checkpoint_loaded") is not False:
        raise ValueError(
            "Pole checkpoint lineage does not prove the source final actor was never loaded"
        )

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT
    source_manifest_path = (root / config["source_manifest"]).resolve()
    report = verify_source_manifest(source_manifest_path, purpose="runtime")
    manifest = json.loads(source_manifest_path.read_text())
    recipe = resolve_source_recipe(config_path, project_root=root, purpose="runtime")
    if identity.get("source_warmup_checkpoint_sha256") != recipe.warmup_checkpoint_sha256:
        raise ValueError("Pole checkpoint was initialized from a different source warmup")
    if identity.get("source_revision") != recipe.source_revision:
        raise ValueError("Pole checkpoint source revision differs from the registered target")
    if identity.get("source_final_checkpoint_sha256") != recipe.final_checkpoint_sha256:
        raise ValueError("Pole checkpoint records a different source final reference")
    if identity.get("robot_xml_sha256") != manifest["robot_contract"]["xml_sha256"]:
        raise ValueError("Pole checkpoint robot XML identity differs from the source contract")
    data_manifest = (root / config["manifest"]).resolve()
    if identity.get("data_manifest_sha256") != _sha256(data_manifest):
        raise ValueError("Pole checkpoint was trained on a different paired data manifest")

    robot_xml = source_robot_xml(source_manifest_path)
    robot = SourcePoleRobot(robot_xml)
    actor, _, _ = build_source_models(robot)
    try:
        actor.load_state_dict(saved["actor"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(
            "Pole checkpoint actor state differs from the source profile contract"
        ) from exc
    if actor.decoder[-1].out_features != 22:
        raise ValueError("Pole checkpoint action dimension is not 22")
    actor.to(device).eval()
    provenance = dict(identity)
    provenance.update(
        {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": checkpoint_format,
            "source_manifest": str(source_manifest_path),
            "source_verified_artifacts": report["verified_artifacts"],
            "source_actor_parity_tolerance": SOURCE_ACTOR_PARITY_TOLERANCE,
        }
    )
    return actor, provenance


__all__ = ["load_pole_actor"]
