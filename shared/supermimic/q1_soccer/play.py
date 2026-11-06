"""GUI playback of a trained SuperMimic soccer student using native MuJoCo."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml

from expert.control import BoundedSoccerPlayer
from expert.play import run
from expert.runtime import ROOT as EXPERT_ROOT

from .models import VisionPolicy, RuntimePolicy


class StudentPlayback:
    """Load only the student actor, never falling back to expert weights."""

    def __init__(self, checkpoint: Path, seed=42, reset_mode="reference"):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(saved, Mapping)
            or saved.get("format") != "primebot_supermimic_soccer_v1"
        ):
            raise ValueError(
                "Expected a SuperMimic soccer student checkpoint (format=primebot_supermimic_soccer_v1)"
            )
        if not isinstance(saved.get("actor"), Mapping):
            raise ValueError("Checkpoint is missing the student actor weights")
        cfg = yaml.safe_load((EXPERT_ROOT / "configs/Q1.yaml").read_text())
        policy = VisionPolicy(cfg)
        policy.load_state_dict(saved["actor"], strict=True)
        policy.eval().requires_grad_(False)
        self.player = BoundedSoccerPlayer(
            seed=seed, reset_mode=reset_mode, policy=RuntimePolicy(policy)
        )
        print(
            f"SuperMimic student: {checkpoint} (iteration {saved.get('iteration', 'unknown')}); inference on CPU",
            flush=True,
        )

    def run(
        self,
        *,
        headless=False,
        steps=None,
        out_video=None,
        show_frustum=True,
        frustum_range=2.5,
    ):
        return run(
            headless=headless,
            steps=steps,
            out_video=out_video,
            show_frustum=show_frustum,
            frustum_range=frustum_range,
            sim=self.player,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
        help="SuperMimic soccer student checkpoint (.ckpt/.pt); GUI opens by default",
    )
    parser.add_argument("--task", choices=["Q1"], default="Q1")
    parser.add_argument("--headless", action="store_true", help="Run without the GUI")
    parser.add_argument(
        "--steps",
        type=int,
        help="Stop after this many physics steps; GUI otherwise runs until closed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset-mode", choices=["reference", "default"], default="reference"
    )
    parser.add_argument(
        "--out-video", type=Path, help="Directory for per-episode MP4 files"
    )
    parser.add_argument(
        "--no-frustum",
        action="store_true",
        help="Hide camera field of view; F toggles it in the GUI",
    )
    parser.add_argument("--frustum-range", type=float, default=2.5)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.frustum_range <= 0:
        parser.error("--frustum-range must be positive")
    torch.set_num_threads(2)
    try:
        playback = StudentPlayback(
            args.checkpoint, seed=args.seed, reset_mode=args.reset_mode
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    playback.run(
        headless=args.headless,
        steps=args.steps,
        out_video=args.out_video,
        show_frustum=not args.no_frustum,
        frustum_range=args.frustum_range,
    )


if __name__ == "__main__":
    main()
