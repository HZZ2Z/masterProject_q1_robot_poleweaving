"""Standalone pole playback; headless by default, optional MuJoCo GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ROOT
from .inference import load_pole_actor
from .source_environment import SourcePoleEnvironment
from .source_playback import load_course_bundle
from .source_runtime import run_source_episode, source_playback_viewer


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
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="连续播放次数（每轮使用递增的随机种子）",
    )
    parser.add_argument("--gui", action="store_true", help="打开 MuJoCo 仿真窗口")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be at least 1")
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
    scene_path = output_dir / "pole-playback.xml"
    environment = SourcePoleEnvironment(case, scene_path, robot, seed=args.seed)
    episodes = []
    with source_playback_viewer(environment, args.gui) as viewer:
        for index in range(args.loops):
            seed = args.seed + index
            episode = run_source_episode(
                actor,
                robot,
                motions,
                case,
                clip,
                scene_path,
                start_frame=0,
                seed=seed,
                device=args.device,
                environment=environment,
                viewer=viewer,
            )
            episodes.append({**episode.result, "seed": seed})
            if viewer is not None and not viewer.is_running():
                break
    result = {
        "checkpoint": identity,
        "case": args.case,
        "loops": len(episodes),
        "requested_loops": args.loops,
        "episodes": episodes,
        "success": all(row["success"] for row in episodes),
        "gui": args.gui,
    }
    if args.loops == 1:
        result.update(episodes[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
