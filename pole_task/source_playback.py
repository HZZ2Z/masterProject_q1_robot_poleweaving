"""Shared paired-course loading for pole evaluation and headless playback."""

from __future__ import annotations

import json
from pathlib import Path

from . import ROOT
from .data import CourseCase, PoleMotions, write_vendor_motion
from .source import source_robot_xml
from .source_profile import SourcePoleRobot
from .source_training import load_training_cases


def load_course_bundle(
    config_path: Path = ROOT / "pole_task/config.json",
    reference_dir: Path | None = None,
    *,
    device: str = "cpu",
) -> tuple[SourcePoleRobot, PoleMotions, dict[str, CourseCase], dict]:
    """Load the registered paired references without any policy weights."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = config_path.parents[1]
    manifest_path = (root / config["manifest"]).resolve()
    source_manifest_path = (root / config["source_manifest"]).resolve()
    training, evaluation, all_cases = load_training_cases(
        config, manifest_path=manifest_path
    )
    robot = SourcePoleRobot(source_robot_xml(source_manifest_path)).to(device)
    reference_dir = Path(reference_dir or root / "logs/q1-pole-weaving/references")
    paths = [
        write_vendor_motion(case, Path(reference_dir) / f"{case.case_id}.npz")
        for case in all_cases
    ]
    motions = PoleMotions(robot, all_cases, paths, device=device)
    motions.prepare()
    cases = {case.case_id: case for case in all_cases}
    return robot, motions, cases, config


__all__ = ["load_course_bundle"]
