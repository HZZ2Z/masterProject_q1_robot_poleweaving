"""Headless fixed-set evaluation for a project-trained pole checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ROOT
from .inference import load_pole_actor
from .source_playback import load_course_bundle
from .source_runtime import evaluate_source_cases
from .source_training import evaluation_cases_for, source_acceptance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    actor, identity = load_pole_actor(args.checkpoint, args.device)
    output_dir = args.output.parent if args.output else ROOT / "logs/q1-pole-weaving/evaluations"
    robot, motions, cases, config = load_course_bundle(
        ROOT / "pole_task/config.json",
        output_dir / "references",
        device=args.device,
    )
    ordered = evaluation_cases_for(config, list(cases.values()))
    reports = evaluate_source_cases(
        actor,
        robot,
        motions,
        ordered,
        output_dir / "pole-evaluation.xml",
        device=args.device,
        seed=args.seed,
    )
    accepted = source_acceptance(reports, config)
    result = {
        "checkpoint": identity,
        "seed": args.seed,
        "cases": reports,
        "success": accepted,
    }
    output_path = args.output or output_dir / f"{Path(identity['checkpoint_path']).stem}-evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
