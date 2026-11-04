"""Course/reference pairing and split checks against source task 7.7."""

import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from pole_task import ROOT
from pole_task.data import (
    COURSE_TRANSLATION,
    load_course,
    load_manifest,
    load_retention,
)
from pole_task.source import verify_source_manifest
from pole_task.training import case_from_retention, select_training_cases

MANIFEST_PATH = ROOT / "pole_task/source_manifest.json"
DATA_MANIFEST = ROOT / "pole_task/data/manifest.json"
COURSE_DIR = ROOT / "pole_task/data/course"
CONFIG = json.loads((ROOT / "pole_task/config.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifacts() -> dict[str, str]:
    return verify_source_manifest(MANIFEST_PATH)["artifact_paths"]


def test_copied_course_catalog_and_geometry_match_registered_source_bytes():
    artifacts = _source_artifacts()
    manifest = load_manifest()

    assert COURSE_DIR.joinpath("source-catalog-v7.toml").read_bytes() == Path(
        artifacts["course_catalog"]
    ).read_bytes()
    assert COURSE_DIR.joinpath("geometry.toml").read_bytes() == Path(
        artifacts["course_geometry"]
    ).read_bytes()
    assert _sha256(COURSE_DIR / "source-catalog-v7.toml") == manifest["source_catalog_sha256"] == _sha256(
        Path(artifacts["course_catalog"])
    )
    assert _sha256(COURSE_DIR / "geometry.toml") == manifest["source_geometry_sha256"] == _sha256(
        Path(artifacts["course_geometry"])
    )


@pytest.mark.parametrize("case_id", ["nominal-01", "eval-01"])
def test_reference_layout_order_and_translation_match_the_paired_course(case_id):
    manifest = load_manifest()
    row = manifest["cases"][case_id]
    geometry = tomllib.loads((COURSE_DIR / "geometry.toml").read_text())["course"]

    assert geometry["origin_xyz_m"] == [0.0, 0.0, 0.0]
    assert geometry["yaw_rad"] == 0.0

    reference_path = DATA_MANIFEST.parent / row["file"]
    with np.load(reference_path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        layout = np.asarray(data["synthesis_layout_xy_m"], dtype=np.float32)

    assert qpos.shape == (row["frames"], 29)
    assert layout.shape == (3, 2)
    assert np.array_equal(layout, np.asarray(row["poles_xy_m"], dtype=np.float32))
    assert np.all(np.diff(layout[:, 0]) > 0), "poles must be ordered along the route"
    spacing = np.diff(layout[:, 0])
    assert np.all(spacing >= geometry["spacing_bounds_m"][0])
    assert np.all(spacing <= geometry["spacing_bounds_m"][1])
    lateral = np.diff(layout[:, 1])
    assert np.all(np.abs(lateral) <= geometry["lateral_offset_bounds_m"][1])

    case = load_course(case_id)
    translated = layout + COURSE_TRANSLATION
    assert np.allclose(case.poles_xy, translated)
    assert np.allclose(case.qpos[:, :2], qpos[:, :2] + COURSE_TRANSLATION)
    assert np.array_equal(case.qpos[:, 2:], qpos[:, 2:])
    assert case.control_steps == row["frames"] - 1


def test_nominal_and_held_out_splits_match_registered_acceptance_steps():
    manifest = load_manifest()
    artifacts = _source_artifacts()
    result = json.loads(Path(artifacts["passed_result"]).read_text())["gate"]

    assert manifest["cases"]["nominal-01"]["split"] == "train"
    assert manifest["cases"]["eval-01"]["split"] == "held_out"
    assert load_course("nominal-01").control_steps == result["nominal_01"]["played_steps"] == 675
    assert load_course("eval-01").control_steps == result["eval_01"]["played_steps"] == 671

    for name, steps in (("straight", 198), ("left_turn", 298), ("right_turn", 298)):
        assert len(load_retention(name)) - 1 == result["retention"][name]["steps"] == steps


def test_training_case_selection_never_accepts_held_out_data():
    nominal = load_course("nominal-01")
    heldout = load_course("eval-01")
    retention = [case_from_retention(name) for name in CONFIG["retention_case_ids"]]
    cases = [nominal, heldout, *retention]

    selected = select_training_cases(cases, CONFIG)
    assert [case.case_id for case in selected] == CONFIG["training_case_ids"]
    assert heldout not in selected
    assert all(case.split != "held_out" for case in selected)

    assert set(CONFIG["training_case_ids"]).isdisjoint(CONFIG["evaluation_case_ids"])
    assert "eval-01" in CONFIG["evaluation_case_ids"]

    with pytest.raises(ValueError, match="Held-out pole cases cannot enter training data"):
        select_training_cases(
            cases, {"training_case_ids": ["nominal-01", "eval-01"]}
        )
    with pytest.raises(ValueError, match="Unknown pole training cases"):
        select_training_cases(cases, {"training_case_ids": ["missing-case"]})
