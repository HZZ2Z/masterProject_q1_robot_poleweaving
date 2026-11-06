"""Paired pole course/reference data and robot-frame task features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from supermimic.data import Q1Motions

from . import DATA

COURSE_TRANSLATION = np.array([-2.0, 2.4], dtype=np.float32)
REFERENCE_WINDOW = 10


@dataclass(frozen=True)
class CourseCase:
    case_id: str
    split: str
    poles_xy: np.ndarray
    qpos: np.ndarray
    path: Path
    sha256: str

    @property
    def control_steps(self) -> int:
        return len(self.qpos) - 1

    @property
    def identity(self) -> str:
        return f"{self.case_id}:{self.sha256}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DATA / "manifest.json") -> dict:
    return json.loads(path.read_text())


def load_course(case_id: str, *, manifest_path: Path = DATA / "manifest.json") -> CourseCase:
    manifest = load_manifest(manifest_path)
    try:
        row = manifest["cases"][case_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pole course case: {case_id}") from exc
    root = manifest_path.parent
    reference = (root / row["file"]).resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Missing paired reference for {case_id}: {reference}")
    digest = _sha256(reference)
    if digest != row["sha256"]:
        raise ValueError(f"Reference hash mismatch for paired course {case_id}")
    with np.load(reference, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32).copy()
    if qpos.ndim != 2 or qpos.shape[1] != 29:
        raise ValueError(f"{case_id} must contain Q1 V3 qpos frames with 29 values")
    if len(qpos) != row["frames"] or len(qpos) - 1 != row["control_steps"]:
        raise ValueError(f"Reference frame count mismatch for {case_id}")
    poles = np.asarray(row["poles_xy_m"], dtype=np.float32)
    if poles.shape != (3, 2):
        raise ValueError(f"{case_id} must pair exactly three ordered poles")
    poles = poles + COURSE_TRANSLATION
    qpos[:, :2] += COURSE_TRANSLATION
    return CourseCase(case_id, row["split"], poles, qpos, reference, digest)


def load_retention(name: str, *, manifest_path: Path = DATA / "manifest.json") -> np.ndarray:
    manifest = load_manifest(manifest_path)
    try:
        row = manifest["retention"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown retention reference: {name}") from exc
    path = (manifest_path.parent / row["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing retention reference {name}: {path}")
    if _sha256(path) != row["sha256"]:
        raise ValueError(f"Retention reference hash mismatch: {name}")
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32).copy()
    if len(qpos) != row["frames"] or qpos.shape[1] != 29:
        raise ValueError(f"Retention reference shape mismatch: {name}")
    qpos[:, :2] += COURSE_TRANSLATION
    return qpos


def policy_course_features(
    reference_window: Tensor,
    poles_xy: Tensor,
    root_xy: Tensor,
    yaw: Tensor,
) -> Tensor:
    """Encode ten future Q1 poses and ordered pole centers in the robot frame."""
    if reference_window.ndim != 3 or reference_window.shape[1:] != (REFERENCE_WINDOW, 29):
        raise ValueError("reference_window must have shape (batch, 10, 29)")
    if poles_xy.shape[-2:] != (3, 2):
        raise ValueError("poles_xy must have shape (batch, 3, 2)")
    c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)

    def to_robot_frame(delta: Tensor) -> Tensor:
        return torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1)

    rel_root = to_robot_frame(reference_window[..., :2] - root_xy[:, None, :])
    # Q1 root quaternions use WXYZ order.
    rq = reference_window[..., 3:7]
    ref_yaw = torch.atan2(
        2 * (rq[..., 0] * rq[..., 3] + rq[..., 1] * rq[..., 2]),
        1 - 2 * (rq[..., 2].square() + rq[..., 3].square()),
    )
    rel_yaw = ref_yaw - yaw[:, None]
    ref = torch.cat(
        (
            rel_root,
            reference_window[..., 2:3],
            rel_yaw.sin().unsqueeze(-1),
            rel_yaw.cos().unsqueeze(-1),
            reference_window[..., 7:29],
        ),
        -1,
    ).flatten(1)
    poles_local = to_robot_frame(poles_xy - root_xy[:, None, :])
    # Retention clips use x=100 as an explicit no-course-poles sentinel.
    has_poles = poles_xy[:, 0, 0] < 50.0
    poles_local = torch.where(has_poles[:, None, None], poles_local, torch.zeros_like(poles_local))
    poles = poles_local.flatten(1)
    return torch.cat((ref, poles), -1)


def features_at(case: CourseCase, frame: int, root_xy: Tensor, yaw: Tensor) -> Tensor:
    """Build one policy feature row, clamping the future window at the endpoint."""
    if not 0 <= frame < len(case.qpos):
        raise ValueError(f"Reference frame {frame} is outside {case.case_id}")
    indexes = np.minimum(np.arange(frame + 1, frame + 1 + REFERENCE_WINDOW), len(case.qpos) - 1)
    window = torch.as_tensor(case.qpos[indexes], dtype=root_xy.dtype, device=root_xy.device).unsqueeze(0)
    poles = torch.as_tensor(case.poles_xy, dtype=root_xy.dtype, device=root_xy.device).unsqueeze(0)
    return policy_course_features(window, poles, root_xy.reshape(1, 2), yaw.reshape(1)).squeeze(0)


def write_vendor_motion(case: CourseCase, output_path: Path) -> Path:
    """Write a translated Q1 reference in the pinned vendor's NPZ format."""
    if case.path is None or not case.path.is_file():
        raise FileNotFoundError(f"Missing source motion for course {case.case_id}")
    with np.load(case.path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["qpos"] = case.qpos
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


class PoleMotions:
    """Course-paired wrapper around vendor Q1Motions reference storage."""

    def __init__(self, robot, cases: list[CourseCase], paths: list[Path], *, device="cpu"):
        if not cases or len(cases) != len(paths):
            raise ValueError("Pole motion cases and vendor paths must be nonempty and paired")
        self.robot = robot
        self.cases = tuple(cases)
        self.dt = 0.02
        self.base = Q1Motions(robot, paths, dt=self.dt, device=device)
        self.poles_xy = torch.as_tensor(np.stack([case.poles_xy for case in cases]), dtype=torch.float32)
        self.paths = tuple(Path(path).resolve() for path in paths)

    def prepare(self) -> None:
        self.base.prepare()

    def lengths(self) -> Tensor:
        return self.base.lengths()

    def remaining(self, clips: Tensor, frames: Tensor) -> Tensor:
        return self.base.remaining(clips, frames)

    def sequence(self, clips: Tensor, frames: Tensor, offsets: Tensor) -> Tensor:
        sequence = self.base.sequence(clips, frames, offsets)
        caller = clips.device
        paired_poles = self.poles_xy.to(device=caller)[clips]
        paired_poles = paired_poles[:, None, :, :].expand(-1, sequence.shape[1], -1, -1)
        return torch.cat((sequence, paired_poles.flatten(start_dim=-2)), -1)
