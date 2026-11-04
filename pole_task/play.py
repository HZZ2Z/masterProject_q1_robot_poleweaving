"""Standalone pole playback; headless by default, optional MuJoCo GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ROOT
from .inference import load_pole_actor
from .source_playback import load_course_bundle
from .source_runtime import run_source_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("nominal-01", "eval-01", "straight", "left_turn", "right_turn"),
        default="nominal-01",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--gui", action="store_true", help="Open the optional MuJoCo viewer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    actor, identity = load_pole_actor(args.checkpoint, args.device)
    output = args.output or ROOT / "logs/q1-pole-weaving/playback.json"
    output_dir = output.parent
    robot, motions, cases, _ = load_course_bundle(
        ROOT / "pole_task/config.json",
        output_dir / "references",
        device=args.device,
    )
    if args.case not in cases:
        raise ValueError(f"Pole playback case {args.case!r} is not registered")
    case = cases[args.case]
    clip = next(
        index for index, motion_case in enumerate(motions.cases)
        if motion_case.case_id == args.case
    )
    episode = run_source_episode(
        actor,
        robot,
        motions,
        case,
        clip,
        output_dir / "pole-playback.xml",
        start_frame=0,
        seed=args.seed,
        device=args.device,
        show_gui=args.gui,
    )
    result = {
        "checkpoint": identity,
        **episode.result,
        "seed": args.seed,
        "gui": args.gui,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
