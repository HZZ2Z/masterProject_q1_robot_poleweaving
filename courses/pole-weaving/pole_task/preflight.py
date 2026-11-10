"""Read-only preflight for pole data, Q1 identity, and local expert weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from . import COURSE_ROOT, ROOT, project_path
from .data import load_course, load_manifest, load_retention
from .source import verify_source_manifest
from .source_checkpoint import load_source_pole
from .source_recipe import resolve_source_recipe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    root = ROOT
    manifest_path = project_path(config["manifest"]).resolve()
    source_manifest_path = project_path(config.get(
        "source_manifest", "courses/pole-weaving/pole_task/source_manifest.json"
    )).resolve()
    source = verify_source_manifest(source_manifest_path)
    warmup_path = Path(source["artifact_paths"][source["warmup_checkpoint_id"]])
    output_root = (root / config["output_root"]).resolve()
    if output_root == (root / "logs").resolve() or root not in output_root.parents:
        raise ValueError("Pole run output must stay in a dedicated project logs directory")
    soccer_outputs = [root / "logs/auto", root / "logs/latest", root / "logs/q1"]
    for soccer_output in soccer_outputs:
        soccer_output = soccer_output.resolve()
        if output_root == soccer_output or output_root in soccer_output.parents or soccer_output in output_root.parents:
            raise ValueError(f"Pole run output overlaps the soccer output path: {soccer_output}")

    manifest = load_manifest(manifest_path)
    course_root = manifest_path.parent / "course"
    source_catalog = course_root / "source-catalog-v7.toml"
    source_geometry = course_root / "geometry.toml"
    if _sha256(source_catalog) != manifest["source_catalog_sha256"]:
        raise ValueError("Copied source catalog hash differs from task 7.7 provenance")
    if _sha256(source_geometry) != manifest["source_geometry_sha256"]:
        raise ValueError("Copied source pole geometry hash differs from task 7.7 provenance")
    local_catalog = tomllib.loads((course_root / "catalog.toml").read_text())
    local_rows = {row["id"]: row for row in local_catalog.get("case", [])}
    if set(local_rows) != set(manifest["cases"]):
        raise ValueError("Local catalog cases differ from reference manifest")
    for case_id, row in local_rows.items():
        paired = manifest["cases"][case_id]
        if row["split"] != paired["split"] or row["motion"] != f"../{paired['file']}":
            raise ValueError(f"Local course catalog does not match paired reference {case_id}")
        if row["poles_xy_m"] != paired["poles_xy_m"]:
            raise ValueError(f"Local pole layout does not match reference {case_id}")
    cases = {}
    retention_ids = set(config["retention_case_ids"])
    training_ids = set(config["training_case_ids"])
    unknown_training = training_ids - set(manifest["cases"]) - retention_ids
    if unknown_training:
        raise ValueError(f"Unknown pole training cases: {sorted(unknown_training)}")
    for case_id in [
        case_id
        for case_id in config["training_case_ids"] + config["evaluation_case_ids"]
        if case_id in manifest["cases"]
    ]:
        case = load_course(case_id, manifest_path=manifest_path)
        expected_split = "train" if case_id in config["training_case_ids"] else "held_out"
        if case.split != expected_split:
            raise ValueError(f"Course {case_id} is paired with split {case.split}, expected {expected_split}")
        cases[case_id] = {
            "identity": case.identity,
            "split": case.split,
            "control_steps": case.control_steps,
            "poles_xy_m": case.poles_xy.tolist(),
        }
    retention = {}
    for name in config["retention_case_ids"]:
        reference = load_retention(name, manifest_path=manifest_path)
        retention[name] = {"frames": len(reference), "control_steps": len(reference) - 1}

    robot_config_path = ROOT / "shared/supermimic/expert/configs/Q1.yaml"
    robot_config = yaml.safe_load(robot_config_path.read_text())
    if robot_config["env"]["num_actions"] != 22 or len(robot_config["control"]["action_scale"]) != 22:
        raise ValueError("Q1 V3 expert config does not define the expected 22 actions")
    scene_source = ROOT / "shared/q1-mujoco/resources/q1_description/q1_soccer.xml"
    xml = ET.parse(scene_source).getroot()
    actuator_names = [node.get("name") for node in xml.findall("./actuator/*")]
    expected_names = robot_config["asset"]["dof_names"]
    if actuator_names != expected_names:
        raise ValueError("Q1 V3 XML actuator ordering differs from expert config")
    for case_id in cases:
        reference_path = manifest_path.parent / manifest["cases"][case_id]["file"]
        with np.load(reference_path, allow_pickle=False) as data:
            if data["root_quat_order"].item().lower() != "wxyz":
                raise ValueError(f"Pole reference {case_id} does not use MuJoCo WXYZ root quaternions")
            if data["robot_joint_names"].tolist() != expected_names:
                raise ValueError(f"Pole reference joint order differs from Q1 V3: {case_id}")
    for name, row in manifest["retention"].items():
        reference_path = manifest_path.parent / row["file"]
        with np.load(reference_path, allow_pickle=False) as data:
            if data["robot_joint_names"].tolist() != expected_names:
                raise ValueError(f"Retention joint order differs from Q1 V3: {name}")

    requested_device = str(config.get("device", "cpu"))
    hardware = {
        "cpu_count": __import__("os").cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "requested_device": requested_device,
        "ready": requested_device == "cpu",
    }
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(requested_device)
            minimum = float(config.get("gpu_min_free_gb", 0)) * 1024**3
            capability = torch.cuda.get_device_capability(requested_device)
            architecture = f"sm_{capability[0]}{capability[1]}"
            supported_arches = torch.cuda.get_arch_list()
            compiled_capabilities = [
                int(name.removeprefix("sm_"))
                for name in supported_arches
                if name.startswith("sm_")
            ]
            architecture_supported = bool(compiled_capabilities) and (
                capability[0] * 10 + capability[1] <= max(compiled_capabilities)
            )
            hardware.update(
                gpu_free_gb=free_bytes / 1024**3,
                gpu_total_gb=total_bytes / 1024**3,
                gpu_min_free_gb=minimum / 1024**3,
                gpu_name=torch.cuda.get_device_name(requested_device),
                gpu_capability=architecture,
                torch_cuda_arches=supported_arches,
                gpu_architecture_supported=architecture_supported,
                ready=free_bytes >= minimum and architecture_supported,
            )
        else:
            hardware["ready"] = False
    elif requested_device != "cpu":
        hardware["ready"] = False
        hardware["device_error"] = "device must be 'cpu' or an available CUDA device"

    recipe = resolve_source_recipe(config_path, project_root=root)
    # Hash verification above only proves the artifacts are byte-identical.
    # Strict model construction and state-dict loading prove the warmup can
    # initialize this profile; failures stop here, before any run directory.
    with torch.random.fork_rng(devices=[]):
        loaded = load_source_pole(source_manifest_path, device="cpu")
    initialization = {
        "source_checkpoint_id": source["warmup_checkpoint_id"],
        "required_path": str(warmup_path),
        "available": warmup_path.is_file(),
        "sha256": recipe.warmup_checkpoint_sha256,
        "strictly_loaded": True,
        "warmup_iteration": loaded.provenance["warmup_iteration"],
        "source_revision": recipe.source_revision,
        "source_config": loaded.provenance["source_config"],
        "source_config_sha256": loaded.provenance["source_config_sha256"],
        "final_checkpoint_id": recipe.final_checkpoint_id,
        "final_checkpoint_sha256": recipe.final_checkpoint_sha256,
        "source_final_checkpoint_loaded": False,
        "state_dict_mapping": loaded.provenance["state_dict_mapping"],
        "provenance": loaded.provenance,
    }
    report = {
        "data": {"manifest": str(manifest_path), "cases": cases, "retention": retention},
        "q1_v3": {
            "scene_sha256": _sha256(scene_source),
            "joint_names": actuator_names,
            "action_dim": len(actuator_names),
        },
        "output_root": str(output_root),
        "hardware": hardware,
        "source": source,
        "recipe": recipe.to_dict(),
        "initialization": initialization,
        "training_ready": True,
        "training_blocker": None,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=COURSE_ROOT / "pole_task/config.json")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config)
    except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"preflight": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return (
        0
        if report["training_ready"]
        and report["initialization"]["available"]
        and report["hardware"]["ready"]
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
