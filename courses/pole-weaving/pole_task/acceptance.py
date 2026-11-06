"""Headless source-aligned acceptance for a project pole checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ROOT
from .inference import load_pole_actor
from .source_playback import load_course_bundle
from .source_runtime import evaluate_source_cases, run_source_episode
from .source_training import evaluation_cases_for, source_acceptance

DEFAULT_TRIALS = 20


def nominal_trial_passes(row: dict, config: dict) -> bool:
    """Source nominal acceptance for one deterministic playback trial."""
    return bool(
        row["route_completed"]
        and not row["pole_collision"]
        and not row["fall"]
        and row["endpoint_distance_m"] <= float(config.get("course_success_endpoint_m", 0.5))
        and row["root_xy_rmse_m"] <= float(config.get("course_success_rmse_m", 0.5))
    )


def run_nominal_trials(
    actor,
    robot,
    motions,
    case,
    scene_path,
    *,
    config: dict,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict]:
    """Run fixed-seed nominal playbacks and record every outcome."""
    if trials <= 0:
        raise ValueError("At least one nominal trial is required")
    index = next(
        (
            position
            for position, motion_case in enumerate(motions.cases)
            if motion_case.identity == case.identity
        ),
        None,
    )
    if index is None:
        raise ValueError("Nominal trial case is absent from the paired motion library")
    rows = []
    for trial in range(trials):
        trial_seed = seed + trial
        episode = run_source_episode(
            actor,
            robot,
            motions,
            case,
            index,
            scene_path,
            start_frame=0,
            seed=trial_seed,
            device=device,
            observation_noise=False,
            action_noise=0.0,
        )
        row = {
            **episode.result,
            "trial": trial,
            "seed": trial_seed,
        }
        row["passed"] = nominal_trial_passes(row, config)
        rows.append(row)
    return rows


def full_acceptance(
    actor,
    robot,
    motions,
    cases,
    scene_path,
    *,
    config: dict,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Evaluate every registered case and require all nominal trials to pass."""
    ordered = evaluation_cases_for(config, list(cases.values()) if isinstance(cases, dict) else cases)
    reports = evaluate_source_cases(
        actor,
        robot,
        motions,
        ordered,
        scene_path,
        device=device,
        seed=seed,
    )
    nominal = next(case for case in ordered if case.case_id == "nominal-01")
    trial_rows = run_nominal_trials(
        actor,
        robot,
        motions,
        nominal,
        scene_path,
        config=config,
        trials=trials,
        seed=seed,
        device=device,
    )
    return {
        "cases": reports,
        "nominal_trials": trial_rows,
        "all_trials_passed": all(row["passed"] for row in trial_rows),
        "success": source_acceptance(reports, config)
        and all(row["passed"] for row in trial_rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    actor, identity = load_pole_actor(args.checkpoint, args.device)
    output_dir = (
        args.output.parent
        if args.output
        else ROOT / "logs/q1-pole-weaving/acceptance"
    )
    robot, motions, cases, config = load_course_bundle(
        ROOT / "pole_task/config.json",
        output_dir / "references",
        device=args.device,
    )
    result = {
        "checkpoint": identity,
        "seed": args.seed,
        "trials": args.trials,
        **full_acceptance(
            actor,
            robot,
            motions,
            cases,
            output_dir / "pole-acceptance.xml",
            config=config,
            trials=args.trials,
            seed=args.seed,
            device=args.device,
        ),
    }
    output_path = (
        args.output
        or output_dir / f"{Path(identity['checkpoint_path']).stem}-acceptance.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
