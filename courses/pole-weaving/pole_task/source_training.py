"""Source task 7.7 pole-profile training, resume, and checkpoint selection.

Training starts only from the registered source warmup checkpoint.  Every run
records the source revision, warmup hash, resolved recipe, data split, and
randomization/actor-stage transition in ``identity.json`` and in each
checkpoint.  A run is bounded by the resolved 2,500-update recipe, an optional
caller limit, and the configured environment-step budget; there is no fallback
to the soccer expert checkpoint or the source task's final actor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import ROOT
from .data import (
    CourseCase,
    PoleMotions,
    load_course,
    load_retention,
    write_vendor_motion,
)
from .preflight import preflight
from .source import verify_source_manifest
from .source_checkpoint import LoadedSourcePole, load_source_pole
from .source_profile import SOURCE_HISTORY_FRAMES, SourcePoleRobot
from .source_recipe import (
    SourceRecipe,
    SourceStage,
    freeze_pole_input_columns,
    resolve_source_recipe,
)
from .source_runtime import (
    SOURCE_REWARD_TUNING,
    SourceEpisode,
    SourcePoleSuperMimic,
    evaluate_source_cases,
    make_source_replay,
    run_source_episode,
)

SOURCE_CHECKPOINT_FORMAT = "supermimic_pole_weaving_profile_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class TrainingBudget:
    """Bounded environment interaction accounting."""

    maximum_steps: int
    steps: int = 0

    def charge(self, steps: int) -> None:
        self.steps += int(steps)
        if self.steps > self.maximum_steps:
            raise RuntimeError(
                f"Pole training exceeded its environment-step budget: "
                f"{self.steps} > {self.maximum_steps}"
            )

    def remaining(self) -> int:
        return max(0, self.maximum_steps - self.steps)


def retention_case(name: str, *, manifest_path: Path) -> CourseCase:
    qpos = load_retention(name, manifest_path=manifest_path)
    path = manifest_path.parent / "references" / f"{name}.npz"
    return CourseCase(
        name,
        "retention",
        np.array([[100, 0], [102, 0], [104, 0]], dtype=np.float32),
        qpos,
        path,
        digest(path),
    )


def load_training_cases(
    config: dict, *, manifest_path: Path
) -> tuple[list[CourseCase], list[CourseCase], list[CourseCase]]:
    """Return (training, evaluation, all) paired cases in catalog order."""
    retention_ids = set(config.get("retention_case_ids", []))
    training: list[CourseCase] = []
    for case_id in config.get("training_case_ids", []):
        if case_id in retention_ids:
            case = retention_case(case_id, manifest_path=manifest_path)
        else:
            case = load_course(case_id, manifest_path=manifest_path)
        if case.split == "held_out":
            raise ValueError(f"Held-out pole case {case_id} cannot enter training data")
        training.append(case)
    evaluation = [
        load_course(case_id, manifest_path=manifest_path)
        for case_id in config.get("evaluation_case_ids", [])
    ]
    held_out = {case.case_id for case in evaluation}
    overlap = held_out.intersection(case.case_id for case in training)
    if overlap:
        raise ValueError(f"Pole cases are both trained and evaluated: {sorted(overlap)}")
    all_cases = training + evaluation
    if not all_cases:
        raise ValueError("Pole training needs at least one registered paired case")
    return training, evaluation, all_cases


def _checkpoint_identity(
    *,
    config: dict,
    config_path: Path,
    recipe: SourceRecipe,
    loaded: LoadedSourcePole,
    manifest_path: Path,
    training: list[CourseCase],
    evaluation: list[CourseCase],
    seed: int,
    device: str,
    resume_checkpoint: Path | None,
) -> dict:
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "profile": "pole-weaving",
        "seed": seed,
        "device": device,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "source_revision": recipe.source_revision,
        "source_warmup_checkpoint_id": recipe.warmup_checkpoint_id,
        "source_warmup_checkpoint_path": loaded.provenance["warmup_checkpoint_path"],
        "source_warmup_checkpoint_sha256": recipe.warmup_checkpoint_sha256,
        "source_final_checkpoint_id": recipe.final_checkpoint_id,
        "source_final_checkpoint_sha256": recipe.final_checkpoint_sha256,
        "source_final_checkpoint_loaded": False,
        "source_config": loaded.provenance["source_config"],
        "source_config_sha256": loaded.provenance["source_config_sha256"],
        "source_state_dict_mapping": loaded.provenance["state_dict_mapping"],
        "robot_xml_sha256": loaded.provenance["robot_xml_sha256"],
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": digest(manifest_path),
        "data_split": {
            "train": {case.case_id: case.identity for case in training},
            "held_out": {case.case_id: case.identity for case in evaluation},
        },
        "recipe": recipe.to_dict(),
        "started_unix": time.time(),
        "resume_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
        "resume_checkpoint_sha256": digest(resume_checkpoint) if resume_checkpoint else None,
    }


def _make_optimizers(algorithm, recipe: SourceRecipe) -> None:
    """Replace vendor defaults with the exact registered Adam settings."""
    for name, settings in (
        ("model", recipe.model_optimizer),
        ("critic", recipe.critic_optimizer),
        ("actor", recipe.actor_optimizer),
    ):
        parameters = getattr(algorithm, f"{name}").parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=settings.learning_rate,
            betas=settings.betas,
            eps=settings.eps,
            weight_decay=settings.weight_decay,
        )
        setattr(algorithm, f"{name}_optimizer", optimizer)


def validate_resume_payload(
    payload: dict,
    *,
    config: dict,
    config_path: Path,
    recipe: SourceRecipe,
    manifest_path: Path,
    source_manifest_path: Path,
) -> int:
    """Validate a project-trained checkpoint and return completed updates."""
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint must be a mapping")
    if payload.get("format") != SOURCE_CHECKPOINT_FORMAT:
        raise ValueError(
            "Resume checkpoint is not a project-trained pole-profile checkpoint "
            f"({SOURCE_CHECKPOINT_FORMAT!r})"
        )
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Resume checkpoint is missing its pole-profile identity")
    if identity.get("profile") != "pole-weaving":
        raise ValueError("Resume checkpoint profile is not pole-weaving")
    if identity.get("source_final_checkpoint_loaded") is not False:
        raise ValueError("Resume checkpoint does not prove the source final actor was never loaded")
    if identity.get("source_warmup_checkpoint_sha256") != recipe.warmup_checkpoint_sha256:
        raise ValueError("Resume checkpoint was initialized from a different source warmup")
    if identity.get("source_revision") != recipe.source_revision:
        raise ValueError("Resume checkpoint source revision differs from the registered task")
    source_manifest = json.loads(Path(source_manifest_path).read_text())
    if identity.get("robot_xml_sha256") != source_manifest["robot_contract"]["xml_sha256"]:
        raise ValueError("Resume checkpoint robot XML hash differs from the source contract")
    if identity.get("data_manifest_sha256") != digest(manifest_path):
        raise ValueError("Resume checkpoint was trained on a different pole data manifest")
    if identity.get("config_sha256") != digest(config_path) or payload.get("config") != config:
        raise ValueError("Resume checkpoint configuration differs from the requested pole run")
    if payload.get("recipe") != recipe.to_dict():
        raise ValueError("Resume checkpoint training recipe differs from the registered source recipe")
    update = payload.get("update")
    if not isinstance(update, int) or not 0 <= update <= recipe.total_updates:
        raise ValueError(
            f"Resume update must be an integer in 0..{recipe.total_updates}, got {update!r}"
        )
    stage = payload.get("stage")
    expected = recipe.stage_for_update(min(update + 1, recipe.total_updates)).name
    if stage != expected:
        raise ValueError(
            f"Resume checkpoint stage {stage!r} disagrees with update {update} "
            f"(expected {expected!r})"
        )
    return update


def load_resume_checkpoint(
    path: Path,
    *,
    config: dict,
    config_path: Path,
    recipe: SourceRecipe,
    manifest_path: Path,
    source_manifest_path: Path,
) -> tuple[dict, int]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pole resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    update = validate_resume_payload(
        payload,
        config=config,
        config_path=config_path,
        recipe=recipe,
        manifest_path=manifest_path,
        source_manifest_path=source_manifest_path,
    )
    if update >= recipe.total_updates:
        raise ValueError(
            f"Resume checkpoint has already completed all {recipe.total_updates} updates"
        )
    return payload, update


def checkpoint_payload(
    *,
    algorithm,
    recipe: SourceRecipe,
    config: dict,
    identity: dict,
    update: int,
    stage: SourceStage,
    best_score: float,
    patience: int,
    evaluation: dict | None = None,
) -> dict:
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "profile": "pole-weaving",
        "stage": stage.name,
        "update": int(update),
        "actor": algorithm.actor.state_dict(),
        "model": algorithm.model.state_dict(),
        "critic": algorithm.critic.state_dict(),
        "actor_optimizer": algorithm.actor_optimizer.state_dict(),
        "model_optimizer": algorithm.model_optimizer.state_dict(),
        "critic_optimizer": algorithm.critic_optimizer.state_dict(),
        "config": config,
        "recipe": recipe.to_dict(),
        "identity": identity,
        "best_score": float(best_score),
        "patience": int(patience),
        "evaluation": evaluation,
    }


def checkpoint_output_hashes(run_dir: Path) -> dict[str, str]:
    """Hash every published checkpoint so a run's output is auditable."""
    return {
        f"{name}_checkpoint_sha256": digest(path)
        for name, path in (
            ("best", Path(run_dir) / "best.pt"),
            ("latest", Path(run_dir) / "latest.pt"),
        )
        if path.is_file()
    }


def select_checkpoint(run_dir: Path, *, prefer_best: bool = True) -> Path:
    """Select a project-trained checkpoint without any source/soccer fallback."""
    run_dir = Path(run_dir).resolve()
    names = ("best.pt", "latest.pt") if prefer_best else ("latest.pt", "best.pt")
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No project-trained pole checkpoint (best.pt/latest.pt) in {run_dir}"
    )


def source_acceptance(reports: dict, config: dict) -> bool:
    """Source-aligned task 7.7 acceptance used to select the best checkpoint."""
    endpoint_limit = float(config.get("course_success_endpoint_m", 0.5))
    rmse_limit = float(config.get("course_success_rmse_m", 0.5))
    retention_limit = float(config.get("retention_rmse_m", 0.3))
    for case_id in config.get("evaluation_case_ids", []) + ["nominal-01"]:
        row = reports.get(case_id)
        if row is None:
            return False
        if (
            not row["route_completed"]
            or row["pole_collision"]
            or row["fall"]
            or row["endpoint_distance_m"] > endpoint_limit
            or row["root_xy_rmse_m"] > rmse_limit
        ):
            return False
    for case_id in config.get("retention_case_ids", []):
        row = reports.get(case_id)
        if row is None or row["fall"] or row["root_xy_rmse_m"] > retention_limit:
            return False
    return True


def evaluation_cases_for(config: dict, cases: list[CourseCase]) -> list[CourseCase]:
    """Return every case required by source acceptance in registered order."""
    by_id = {case.case_id: case for case in cases}
    required = [
        "nominal-01",
        *config.get("evaluation_case_ids", []),
        *config.get("retention_case_ids", []),
    ]
    missing = [case_id for case_id in required if case_id not in by_id]
    if missing:
        raise ValueError(f"Acceptance cases are missing: {missing}")
    return [by_id[case_id] for case_id in required]


def _collect_episodes(
    actor,
    robot: SourcePoleRobot,
    motions: PoleMotions,
    cases: list[CourseCase],
    run_dir: Path,
    *,
    stage: SourceStage,
    seed: int,
    device: str,
    episode_counts: dict[str, int],
    action_noise: float,
    budget: TrainingBudget,
) -> list[SourceEpisode]:
    clip_by_case = {case.case_id: index for index, case in enumerate(motions.cases)}
    episodes: list[SourceEpisode] = []
    for case in cases:
        count = int(episode_counts.get(case.case_id, 0))
        if count <= 0:
            continue
        if budget.remaining() <= 0:
            raise RuntimeError("Pole training has no environment-step budget for collection")
        for index in range(count):
            episode = run_source_episode(
                actor,
                robot,
                motions,
                case,
                clip_by_case[case.case_id],
                run_dir / f"collection-{case.case_id}.xml",
                start_frame=None,
                seed=seed + index,
                device=device,
                action_noise=action_noise,
                observation_noise=True,
                clean_policy_actions=True,
                reward_tuning=SOURCE_REWARD_TUNING,
                randomize=stage.randomize,
            )
            budget.charge(episode.steps)
            episodes.append(episode)
    if not episodes:
        raise ValueError("Pole training collection produced no episodes")
    return episodes


def _episode_counts(config: dict, training: list[CourseCase]) -> dict[str, int]:
    nominal = int(config.get("collection_episodes", 1))
    retention = int(config.get("retention_collection_episodes", 1))
    return {
        case.case_id: nominal if case.split == "train" else retention
        for case in training
    }


def _evaluate(
    actor,
    robot: SourcePoleRobot,
    motions: PoleMotions,
    cases: list[CourseCase],
    run_dir: Path,
    *,
    seed: int,
    device: str,
    budget: TrainingBudget,
) -> dict:
    reports = evaluate_source_cases(
        actor,
        robot,
        motions,
        cases,
        run_dir / "evaluation.xml",
        device=device,
        seed=seed,
    )
    budget.charge(sum(int(row["steps"]) for row in reports.values()))
    return reports


def train_source(
    config_path: Path,
    resume_checkpoint: Path | None = None,
    *,
    max_updates: int | None = None,
    episode_counts: dict[str, int] | None = None,
    device: str | None = None,
) -> Path:
    """Train a project pole policy from the registered source warmup.

    ``max_updates`` and ``episode_counts`` exist for bounded smoke runs; the
    production budget remains the resolved source recipe.
    """
    config_path = Path(config_path).resolve()
    report = preflight(config_path)
    if not report["training_ready"]:
        raise ValueError(report["training_blocker"] or "Pole training preflight failed")
    if not report["hardware"]["ready"] and device is None:
        raise RuntimeError(
            "Requested training device failed preflight: "
            + json.dumps(report["hardware"], sort_keys=True)
        )
    config = json.loads(config_path.read_text())
    root = ROOT
    manifest_path = (root / config["manifest"]).resolve()
    source_manifest_path = (root / config["source_manifest"]).resolve()
    recipe = resolve_source_recipe(config_path, project_root=root)
    if max_updates is not None:
        if not 0 < int(max_updates) <= recipe.total_updates:
            raise ValueError(
                f"max_updates must be in 1..{recipe.total_updates}, got {max_updates}"
            )
    # Strictly load after all file/profile validation; no run directory exists
    # yet, so a missing/corrupt source artifact cannot leave a partial run.
    loaded = load_source_pole(source_manifest_path, device="cpu")
    device = device or str(config.get("device", "cpu"))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass a CPU device explicitly")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    robot = loaded.robot.to(device)
    actor = loaded.actor.to(device)
    world = loaded.world_model.to(device)
    critic = loaded.critic.to(device)
    training_cases, evaluation_cases, all_cases = load_training_cases(
        config, manifest_path=manifest_path
    )
    output_root = (root / config["output_root"]).resolve()
    run_dir = output_root / time.strftime("run-%Y%m%d-%H%M%S")
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"{time.strftime('run-%Y%m%d-%H%M%S')}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        motion_dir = run_dir / "references"
        paths = [
            write_vendor_motion(case, motion_dir / f"{case.case_id}.npz")
            for case in all_cases
        ]
        motions = PoleMotions(robot, all_cases, paths, device=device)
        motions.prepare()
        clip_by_case = {case.case_id: index for index, case in enumerate(all_cases)}

        algorithm = SourcePoleSuperMimic(
            robot,
            motions,
            world,
            actor,
            critic,
            learning_rate=recipe.model_optimizer.learning_rate,
            horizon=recipe.imagination_horizon,
            reward_tuning=SOURCE_REWARD_TUNING,
            clean_policy_actions=recipe.clean_policy_actions,
            normalize_policy_objective=True,
            normalize_model_objective=True,
            fuse_model_loss=False,
            cache_reference_heads=False,
            cache_goals=False,
        )
        _make_optimizers(algorithm, recipe)

        resume_payload = None
        completed_updates = 0
        best_score = float("inf")
        patience = 0
        if resume_checkpoint is not None:
            resume_payload, completed_updates = load_resume_checkpoint(
                resume_checkpoint,
                config=config,
                config_path=config_path,
                recipe=recipe,
                manifest_path=manifest_path,
                source_manifest_path=source_manifest_path,
            )
            actor.load_state_dict(resume_payload["actor"], strict=True)
            world.load_state_dict(resume_payload["model"], strict=True)
            critic.load_state_dict(resume_payload["critic"], strict=True)
            algorithm.model_optimizer.load_state_dict(resume_payload["model_optimizer"])
            algorithm.actor_optimizer.load_state_dict(resume_payload["actor_optimizer"])
            algorithm.critic_optimizer.load_state_dict(resume_payload["critic_optimizer"])
            best_score = float(resume_payload.get("best_score", float("inf")))
            patience = int(resume_payload.get("patience", 0))

        identity = _checkpoint_identity(
            config=config,
            config_path=config_path,
            recipe=recipe,
            loaded=loaded,
            manifest_path=manifest_path,
            training=training_cases,
            evaluation=evaluation_cases,
            seed=seed,
            device=device,
            resume_checkpoint=resume_checkpoint,
        )
        identity["resumed_from_update"] = completed_updates
        atomic_json(run_dir / "identity.json", identity)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        budget = TrainingBudget(int(config.get("max_environment_steps", 2_000_000)))
        counts = _episode_counts(config, training_cases)
        if episode_counts:
            counts.update({case_id: int(value) for case_id, value in episode_counts.items()})
        action_noise = float(config.get("action_noise", 0.0))
        next_update = completed_updates + 1
        current_stage = recipe.stage_for_update(
            min(next_update, recipe.total_updates)
        )
        if current_stage.freeze_pole_input_columns:
            freeze_pole_input_columns(actor)
        episodes = _collect_episodes(
            actor,
            robot,
            motions,
            training_cases,
            run_dir,
            stage=current_stage,
            seed=seed + 1000,
            device=device,
            episode_counts=counts,
            action_noise=action_noise,
            budget=budget,
        )
        replay = make_source_replay(episodes, robot, clip_by_case, device=device)
        stop_reason = None
        metrics_path = run_dir / "metrics.jsonl"
        limit = recipe.total_updates if max_updates is None else int(max_updates)
        evaluation_interval = max(1, int(config.get("evaluation_interval", 25)))
        last_update = completed_updates
        for update in range(next_update, limit + 1):
            stage = recipe.stage_for_update(update)
            if stage.name != current_stage.name:
                current_stage = stage
                if stage.freeze_pole_input_columns:
                    freeze_pole_input_columns(actor)
                # Each source stage has its own collection physics; do not fit
                # actor-stage updates on deterministic warmup transitions.
                episodes = _collect_episodes(
                    actor,
                    robot,
                    motions,
                    training_cases,
                    run_dir,
                    stage=stage,
                    seed=seed + 1000 + update,
                    device=device,
                    episode_counts=counts,
                    action_noise=action_noise,
                    budget=budget,
                )
                replay = make_source_replay(episodes, robot, clip_by_case, device=device)
            rates = recipe.optimizer_rates(update - 1)
            for name, rate in rates.items():
                for group in getattr(algorithm, f"{name}_optimizer").param_groups:
                    group["lr"] = rate
            metrics: dict[str, float] = {}
            for _ in range(recipe.model_batches):
                batch = replay.sample(
                    recipe.model_batch_size,
                    window=8,
                    history=1,
                )
                metrics.update(algorithm.fit_model(batch))
            policy_batch = replay.sample(
                recipe.policy_batch_size,
                window=1,
                history=SOURCE_HISTORY_FRAMES,
            )
            metrics.update(
                algorithm.improve_policy(
                    policy_batch,
                    recipe.critic_epochs,
                    recipe.critic_minibatches,
                    update_actor=stage.actor_updates_enabled,
                )
            )
            last_update = update
            metrics.update(
                {
                    "update": update,
                    "stage": stage.name,
                    "actor_updates_enabled": stage.actor_updates_enabled,
                    "randomize": stage.randomize,
                    "model_learning_rate": rates["model"],
                    "actor_learning_rate": rates["actor"],
                    "critic_learning_rate": rates["critic"],
                    "environment_steps": budget.steps,
                }
            )
            should_evaluate = (
                update % evaluation_interval == 0
                or update == recipe.total_updates
                or update == limit
                or stage.end_update == update
            )
            evaluation = None
            if should_evaluate:
                evaluation = _evaluate(
                    actor,
                    robot,
                    motions,
                    evaluation_cases_for(config, all_cases),
                    run_dir,
                    seed=seed + 30000,
                    device=device,
                    budget=budget,
                )
                score = sum(
                    row["root_xy_rmse_m"]
                    for row in evaluation.values()
                    if isinstance(row, dict) and "root_xy_rmse_m" in row
                )
                accepted = source_acceptance(evaluation, config)
                if accepted and score < best_score:
                    best_score, patience = score, 0
                else:
                    patience += 1
                metrics["evaluation"] = evaluation
                metrics["best_score"] = best_score
                metrics["patience"] = patience
                metrics["accepted"] = accepted
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                payload = checkpoint_payload(
                    algorithm=algorithm,
                    recipe=recipe,
                    config=config,
                    identity=identity,
                    update=update,
                    stage=stage,
                    best_score=best_score,
                    patience=patience,
                    evaluation=evaluation,
                )
                torch.save(payload, run_dir / "latest.pt")
                if accepted and score <= best_score:
                    torch.save(payload, run_dir / "best.pt")
                if patience >= int(config.get("success_patience", 100)):
                    stop_reason = {
                        "reason": "success_patience_exhausted",
                        "update": update,
                        "best_score": best_score,
                    }
                    break
            if budget.remaining() <= 0:
                stop_reason = {
                    "reason": "max_environment_steps",
                    "update": update,
                    "environment_steps": budget.steps,
                }
                break
            if max_updates is not None and update >= int(max_updates):
                stop_reason = {
                    "reason": "caller_update_limit",
                    "update": update,
                    "environment_steps": budget.steps,
                }
                break
        if last_update > completed_updates:
            # Always publish the last completed update, even when the final
            # update did not land on an evaluation boundary.
            if not (run_dir / "latest.pt").is_file() or last_update % evaluation_interval:
                payload = checkpoint_payload(
                    algorithm=algorithm,
                    recipe=recipe,
                    config=config,
                    identity=identity,
                    update=last_update,
                    stage=recipe.stage_for_update(last_update),
                    best_score=best_score,
                    patience=patience,
                )
                torch.save(payload, run_dir / "latest.pt")
        stop_reason = stop_reason or {
            "reason": "recipe_exhausted" if last_update == recipe.total_updates else "update_limit",
            "update": last_update,
            "environment_steps": budget.steps,
        }
        atomic_json(run_dir / "stop_reason.json", stop_reason)
        identity.update(
            {
                "completed_updates": last_update,
                "environment_steps": budget.steps,
                "best_checkpoint": str(run_dir / "best.pt")
                if (run_dir / "best.pt").is_file()
                else None,
                "latest_checkpoint": str(run_dir / "latest.pt")
                if (run_dir / "latest.pt").is_file()
                else None,
                "stop_reason": stop_reason,
                **checkpoint_output_hashes(run_dir),
            }
        )
        atomic_json(run_dir / "identity.json", identity)
        return run_dir
    except Exception:
        # Keep the failure visible without publishing a checkpoint or identity
        # that could be mistaken for a completed run.
        atomic_json(
            run_dir / "failure.json",
            {"error": "training failed", "exception": repr(__import__("sys").exc_info()[1])},
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Source-compatible pole training")
    parser.add_argument("--config", type=Path, default=ROOT / "pole_task/config.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.preflight_only:
        try:
            report = preflight(args.config)
        except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as exc:
            print(json.dumps({"preflight": "failed", "error": str(exc)}, indent=2))
            return 2
        print(json.dumps(report, indent=2))
        return 0 if report["training_ready"] and report["initialization"]["available"] else 2
    print(
        train_source(
            args.config,
            args.resume_checkpoint,
            max_updates=args.max_updates,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
