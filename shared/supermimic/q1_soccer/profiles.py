"""Resolve independent training profiles without constructing their models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from . import ROOT


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    config_path: Path
    output_root: Path
    model_contract: str
    checkpoint_format: str


_PROFILE_DEFAULTS = {
    "soccer": {
        "config": "courses/football-kicking/configs/soccer.json",
        "output_key": "output",
        "other_output": "logs/q1-pole-weaving",
        "model_contract": "soccer-vision-79x50-v1",
        "checkpoint_format": "primebot_supermimic_soccer_v1",
    },
    "pole-weaving": {
        "config": "pole_task/config.json",
        "output_key": "output_root",
        "other_output": "logs/auto",
        "model_contract": "source-q1-course-v1",
        "checkpoint_format": "supermimic_pole_weaving_profile_v1",
    },
}


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def resolve_profile(
    name: str = "soccer",
    config_path: str | Path | None = None,
    *,
    project_root: Path = ROOT,
) -> TrainingProfile:
    """Resolve config and output ownership for soccer or pole-weaving."""
    if name not in _PROFILE_DEFAULTS:
        raise ValueError(f"Unknown SuperMimic profile {name!r}; choose soccer or pole-weaving")
    root = Path(project_root).resolve()
    defaults = _PROFILE_DEFAULTS[name]
    resolved_config = _resolve_path(config_path or defaults["config"], root)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"{name} profile config does not exist: {resolved_config}")
    config = json.loads(resolved_config.read_text())
    output_value = config.get(defaults["output_key"])
    if not isinstance(output_value, str) or not output_value:
        raise ValueError(f"{name} profile requires config field {defaults['output_key']!r}")
    output_root = _resolve_path(output_value, root)
    logs_root = (root / "logs").resolve()
    if output_root == logs_root or logs_root not in output_root.parents:
        raise ValueError(f"{name} profile output must stay inside {logs_root}")
    other_output = _resolve_path(defaults["other_output"], root)
    if _overlaps(output_root, other_output):
        other_name = "pole-weaving" if name == "soccer" else "soccer"
        raise ValueError(f"{name} profile output overlaps the {other_name} profile output: {other_output}")

    return TrainingProfile(
        name=name,
        config_path=resolved_config,
        output_root=output_root,
        model_contract=defaults["model_contract"],
        checkpoint_format=defaults["checkpoint_format"],
    )
