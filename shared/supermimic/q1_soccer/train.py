"""Expert-guided soccer training using PrimebotTech's SuperMimic core."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import os
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import signal
import subprocess

import torch

from . import ROOT
from .algorithm import SoccerSuperMimic
from .data import ExpertReplay
from .models import VisionPolicy, SoccerWorldModel, SoccerCritic
from .robot import SoccerRobot
from .rollout import NativeRollouts
from expert.collect import split_for
from .vendor import identity
from .trust import PolicyTrust


@dataclass
class TrainConfig:
    dataset: str = "expert/datasets/q1_bounded_v1"
    checkpoint: str = "expert/checkpoints/model_17800.pth"
    output: str = "logs/auto"
    device: str = "cuda:0"
    iterations: int = 10000
    model_warmup: int = 500
    model_batches: int = 10
    batch_size: int = 128
    horizon: int = 16
    learning_rate: float = 0.0001
    anchor_weight: float = 1.0
    evaluate_every: int = 1
    evaluate_episodes: int = 24
    online_episodes: int = 8
    workers: int = 2
    cpu_threads: int = 2
    gpu_memory_gb: float = 3.0
    gpu_reserve_gb: float = 3.0
    max_world_rmse: float = 0.15
    max_ball_rmse: float = 0.2
    max_success_drop: float = 0.0
    max_fall_increase: float = 0.0
    seed: int = 0
    wandb: bool = False
    warm_start_model: str | None = None
    critic_warmup: int = 100
    actor_learning_rate: float = 1e-6
    bootstrap_weight: float = 0.05
    trust_action_rms: float = 0.01
    trust_action_max: float = 0.05
    max_rejections: int = 0
    reset_actor_optimizer_on_resume: bool = False
    save_every: int = 50
    max_joint_position_rmse: float = 0.06
    max_base_height_rmse: float = 0.02
    max_base_orientation_rmse: float = 0.12
    max_critic_relative_rmse: float = 0.35

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text())
        unknown = set(document) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")
        result = cls(**document)
        for name in ["dataset", "checkpoint", "output"]:
            value = Path(getattr(result, name))
            setattr(result, name, str(value if value.is_absolute() else ROOT / value))
        if Path(result.output).name == "auto":
            name = "q1-robot-supermimic-" + datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).strftime("%Y%m%d-%H%M%S")
            result.output = str(Path(result.output).parent / name)
        if not re.fullmatch(
            r"q1-robot-supermimic-\d{8}-\d{6}", Path(result.output).name
        ):
            raise ValueError(
                "Run directory must be q1-robot-supermimic-<YYYYMMDD-HHMMSS>; use output=logs/auto"
            )
        if result.warm_start_model:
            path = Path(result.warm_start_model)
            result.warm_start_model = str(path if path.is_absolute() else ROOT / path)
        if (
            min(
                result.iterations,
                result.batch_size,
                result.horizon,
                result.model_batches,
                result.evaluate_every,
                result.evaluate_episodes,
                result.workers,
                result.cpu_threads,
            )
            < 1
        ):
            raise ValueError("Counts must be positive")
        if (
            result.model_warmup < 1
            or result.online_episodes < 0
            or result.gpu_memory_gb <= 0
            or result.gpu_reserve_gb < 0
        ):
            raise ValueError("Invalid warmup, rollout count or memory budget")
        if result.evaluate_every != 1:
            raise ValueError(
                "Repaired training requires evaluation after each candidate actor update"
            )
        if (
            min(result.critic_warmup, result.save_every) < 1
            or result.max_rejections < 0
            or min(
                result.actor_learning_rate,
                result.trust_action_rms,
                result.trust_action_max,
            )
            <= 0
        ):
            raise ValueError("Invalid critic warmup or policy trust settings")
        return result


def hardware():
    result = {
        "cpu_count": __import__("os").cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }
    if result["cuda_available"]:
        # nvidia-smi avoids creating a CUDA context for a CPU-only check/worker.
        rows = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
        name, free, total = [item.strip() for item in rows[0].split(",")]
        result.update(
            gpu=name, free_bytes=int(free) * 1024**2, total_bytes=int(total) * 1024**2
        )
    return result


def reserve_device(cfg):
    if cfg.device == "cpu":
        return hardware()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use device=cpu explicitly")
    torch.cuda.set_device(cfg.device)
    free, total = torch.cuda.mem_get_info()
    budget = int(cfg.gpu_memory_gb * 1024**3)
    if free < budget + cfg.gpu_reserve_gb * 1024**3:
        raise RuntimeError(
            f"Insufficient free VRAM: {free / 1024**3:.2f} GiB; need budget {cfg.gpu_memory_gb} + reserve {cfg.gpu_reserve_gb}. Other jobs are left alone."
        )
    torch.cuda.set_per_process_memory_fraction(budget / total, cfg.device)
    return hardware()


def build(cfg, replay):
    identity()
    robot = SoccerRobot().to(cfg.device)
    actor = VisionPolicy(robot.cfg, cfg.checkpoint).to(cfg.device)
    model = SoccerWorldModel(robot).to(cfg.device)
    critic = SoccerCritic(robot).to(cfg.device)
    algorithm = SoccerSuperMimic(
        robot,
        replay,
        model,
        actor,
        critic,
        learning_rate=cfg.learning_rate,
        horizon=cfg.horizon,
        normalize_model_objective=True,
    )
    for group in algorithm.actor_optimizer.param_groups:
        group["lr"] = cfg.actor_learning_rate
    if cfg.warm_start_model:
        source = torch.load(
            cfg.warm_start_model, map_location=cfg.device, weights_only=True
        )
        if source.get("format") != "primebot_supermimic_soccer_v1":
            raise ValueError("Wrong world-model warm-start format")
        if source.get("training_data_policy") != "strict_success_only_v1":
            raise ValueError(
                "World-model warm start must also use strictly successful trajectories only"
            )
        algorithm.model.load_state_dict(source["model"], strict=True)
    return algorithm


def compatible_baseline(baseline, seeds):
    """Only compare scores measured on the same evaluation scenes."""
    return (
        baseline
        if baseline is not None and baseline.get("seeds") == list(seeds)
        else None
    )


def rejection_limit_reached(cfg, count):
    """Zero disables automatic exits; failed candidates are still rolled back."""
    return cfg.max_rejections > 0 and count >= cfg.max_rejections


@torch.no_grad()
def model_validation(algorithm, replay, batch_size=64):
    batch = replay.sample(
        batch_size,
        window=algorithm.horizon,
        device=next(algorithm.model.parameters()).device,
    )
    state = batch.states[:, 0]
    errors = []
    ball_errors = []
    joint_errors = []
    height_errors = []
    orientation_errors = []
    for t in range(batch.actions.shape[1]):
        state = algorithm.model(state, batch.actions[:, t], batch.context)
        target = batch.next_states[:, t]
        aligned = target.clone()
        for section in [slice(3, 7), slice(82, 86), slice(95, 99), slice(105, 109)]:
            aligned[:, section] = torch.where(
                (state[:, section] * target[:, section]).sum(-1, keepdim=True) < 0,
                -target[:, section],
                target[:, section],
            )
        indexes = torch.cat(
            (
                torch.arange(57, device=state.device),
                torch.arange(79, 109, device=state.device),
            )
        )
        errors.append(
            ((state[:, indexes] - aligned[:, indexes]) / algorithm.model.scale[indexes])
            .square()
            .mean()
        )
        ball_errors.append((state[:, 79:82] - target[:, 79:82]).square().mean())
        joint_errors.append((state[:, 7:29] - target[:, 7:29]).square().mean())
        height_errors.append((state[:, 2] - target[:, 2]).square().mean())
        dot = (state[:, 3:7] * target[:, 3:7]).sum(-1).abs().clamp(0, 1)
        orientation_errors.append((2 * torch.acos(dot)).square().mean())
    return dict(
        world_rmse=float(torch.stack(errors).mean().sqrt()),
        ball_position_rmse=float(torch.stack(ball_errors).mean().sqrt()),
        joint_position_rmse=float(torch.stack(joint_errors).mean().sqrt()),
        base_height_rmse=float(torch.stack(height_errors).mean().sqrt()),
        base_orientation_rmse=float(torch.stack(orientation_errors).mean().sqrt()),
        model_validation_steps=algorithm.horizon,
    )


def save(
    path,
    algorithm,
    cfg,
    iteration,
    baseline,
    replay,
    validation,
    online_seed,
    trust=None,
):
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            format="primebot_supermimic_soccer_v1",
            iteration=iteration,
            config=asdict(cfg),
            vendor_commit=identity()["commit"],
            actor=algorithm.actor.state_dict(),
            model=algorithm.model.state_dict(),
            critic=algorithm.critic.state_dict(),
            actor_optimizer=algorithm.actor_optimizer.state_dict(),
            model_optimizer=algorithm.model_optimizer.state_dict(),
            critic_optimizer=algorithm.critic_optimizer.state_dict(),
            baseline=baseline,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if cfg.device.startswith("cuda")
            else [],
            replay_rng=replay.generator.get_state(),
            validation_rng=validation.generator.get_state(),
            online_seed=online_seed,
            online_archives=getattr(replay, "online_archives", []),
            training_revision="real_critic_trust_v2",
            training_data_policy="strict_success_only_v1",
            accepted_policy_updates=trust.accepted_updates if trust else 0,
            consecutive_rejections=trust.consecutive_rejections if trust else 0,
        ),
        temporary,
    )
    temporary.replace(path)


def train(cfg, resume=None):
    resources = reserve_device(cfg)
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    output = Path(cfg.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory, including when resuming")
    replay = ExpertReplay(cfg.dataset, seed=cfg.seed)
    expert_episode_count = len(replay.episodes)
    expert_transition_count = sum(
        len(episode["actions"]) for episode in replay.episodes
    )
    validation = ExpertReplay(cfg.dataset, split="validation", seed=123)
    if (
        hashlib.sha256(Path(cfg.checkpoint).read_bytes()).hexdigest()
        != replay.manifest["checkpoint_sha256"]
    ):
        raise ValueError("Teacher checkpoint differs from dataset")
    algorithm = build(cfg, replay)
    rollouts = NativeRollouts(cfg.workers)
    replay.online_archives = []
    start = 0
    online_seed = 3000000
    baseline = None
    saved = {}
    if resume:
        saved = torch.load(resume, map_location=cfg.device, weights_only=False)
        if saved.get("training_revision") != "real_critic_trust_v2":
            raise ValueError(
                "Use warm_start_model for checkpoints predating this repair"
            )
        if saved.get("training_data_policy") != "strict_success_only_v1":
            raise ValueError(
                "Cannot resume a checkpoint trained with unsuccessful episodes; start fresh"
            )
        for name in ["dataset", "checkpoint", "horizon", "seed"]:
            if saved["config"][name] != getattr(cfg, name):
                raise ValueError(f"Resume configuration differs: {name}")
        for name in [
            "actor",
            "model",
            "critic",
            "actor_optimizer",
            "model_optimizer",
            "critic_optimizer",
        ]:
            if name == "actor_optimizer" and cfg.reset_actor_optimizer_on_resume:
                continue
            getattr(algorithm, name).load_state_dict(saved[name])
        start = saved["iteration"]
        baseline = saved["baseline"]
        online_seed = saved["online_seed"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] and cfg.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        replay.generator.set_state(saved["replay_rng"].cpu())
        validation.generator.set_state(saved["validation_rng"].cpu())
        replay.online_archives = saved.get("online_archives", [])
    elif cfg.warm_start_model:
        source = torch.load(cfg.warm_start_model, map_location="cpu", weights_only=True)
        replay.online_archives = source.get("online_archives", [])
    for archive in replay.online_archives:
        if not Path(archive).is_dir():
            raise FileNotFoundError(f"Missing online replay archive: {archive}")
        for metadata_path in sorted(Path(archive).glob("*.json")):
            import numpy as np

            with np.load(
                metadata_path.with_suffix(".npz"), allow_pickle=False
            ) as shard:
                replay.add(
                    {key: shard[key] for key in shard.files},
                    json.loads(metadata_path.read_text()),
                )
    output.mkdir(parents=True)
    latest = output.parent / "latest"
    temporary_link = output.parent / f".latest-{os.getpid()}"
    temporary_link.symlink_to(output.name, target_is_directory=True)
    temporary_link.replace(latest)
    (output / "data_policy.json").write_text(
        json.dumps(
            {
                "policy": "strict_success_only_v1",
                "expert_training_episodes": expert_episode_count,
                "expert_training_transitions": expert_transition_count,
                "restored_online_training_episodes": len(replay.episodes)
                - expert_episode_count,
                "excluded_unsuccessful_expert_episodes": replay.excluded_unsuccessful,
                "training_transitions": sum(len(e["actions"]) for e in replay.episodes),
                "unsuccessful_training_episodes": sum(
                    not replay.is_success(e["meta"]) for e in replay.episodes
                ),
                "online_unsuccessful": "discarded before replay and disk archive",
                "validation_episodes": len(validation.episodes),
                "validation_used_for_fitting": False,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (output / "hardware.json").write_text(json.dumps(resources, indent=2) + "\n")
    evaluation_seeds = [
        episode["meta"]["seed"]
        for episode in validation.episodes[: cfg.evaluate_episodes]
    ]
    # Appended/cleaned datasets can change held-out reset seeds. Re-evaluate the
    # retained actor instead of comparing scores measured on different scenes.
    baseline = compatible_baseline(baseline, evaluation_seeds)
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    tracker = None
    if cfg.wandb:
        import wandb

        tracker = wandb.init(
            project="Q1-Robot-Soccer",
            group="Primebot-SuperMimic",
            name=output.name,
            config=asdict(cfg),
            dir=str(output),
        )
    if baseline is None:
        baseline = rollouts.evaluate(algorithm.actor, evaluation_seeds)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    trust = PolicyTrust(algorithm, cfg.trust_action_rms, cfg.trust_action_max)
    trust.accepted_updates = saved.get("accepted_policy_updates", 0)
    trust.consecutive_rejections = saved.get("consecutive_rejections", 0)
    if cfg.reset_actor_optimizer_on_resume:
        trust.consecutive_rejections = 0
    save(
        output / "last.pt",
        algorithm,
        cfg,
        start,
        baseline,
        replay,
        validation,
        online_seed,
        trust,
    )
    save(
        output / "best.pt",
        algorithm,
        cfg,
        start,
        baseline,
        replay,
        validation,
        online_seed,
        trust,
    )
    iteration = start
    completed_iteration = start
    try:
        for iteration in range(start + 1, cfg.iterations + 1):
            if stopped:
                break
            record = {"iteration": iteration}
            for _ in range(cfg.model_batches):
                record.update(
                    algorithm.fit_model(
                        replay.sample(
                            cfg.batch_size, window=cfg.horizon, device=cfg.device
                        )
                    )
                )
            validation.generator.manual_seed(123)
            metrics = model_validation(algorithm, validation, min(cfg.batch_size, 64))
            record.update(metrics)
            record.update(
                algorithm.fit_real_critic(
                    replay.sample(cfg.batch_size, window=1, device=cfg.device)
                )
            )
            ready = (
                iteration > cfg.model_warmup
                and iteration > cfg.critic_warmup
                and metrics["world_rmse"] <= cfg.max_world_rmse
                and metrics["ball_position_rmse"] <= cfg.max_ball_rmse
                and metrics["joint_position_rmse"] <= cfg.max_joint_position_rmse
                and metrics["base_height_rmse"] <= cfg.max_base_height_rmse
                and metrics["base_orientation_rmse"] <= cfg.max_base_orientation_rmse
                and record["critic_relative_rmse"] <= cfg.max_critic_relative_rmse
            )
            record["policy_updated"] = ready
            record["stage"] = (
                "policy_and_world_model"
                if ready
                else "world_model_warmup"
                if iteration <= cfg.model_warmup
                else "critic_warmup"
                if iteration <= cfg.critic_warmup
                or record["critic_relative_rmse"] > cfg.max_critic_relative_rmse
                else "awaiting_world_model_validation"
            )
            if ready:
                previous = trust.snapshot()
                guard = replay.sample(
                    max(cfg.batch_size, 512), window=1, device=cfg.device
                )
                with torch.no_grad():
                    previous_actions = algorithm.actor(guard.history).detach()
                batch = replay.sample(
                    cfg.batch_size, window=1, imitation=True, device=cfg.device
                )
                record.update(
                    algorithm.improve_policy(
                        batch,
                        anchor_batch=batch,
                        anchor_weight=cfg.anchor_weight,
                        bootstrap_weight=cfg.bootstrap_weight,
                        fit_critic=False,
                    )
                )
                record.update(
                    trust.constrain(previous, guard.history, previous_actions)
                )
            evaluate_candidate = ready and record.get("trust_passed", False)
            if evaluate_candidate:
                evaluation = rollouts.evaluate(algorithm.actor, evaluation_seeds)
                accepted = (
                    evaluation["verified_success_rate"]
                    >= baseline["verified_success_rate"] - cfg.max_success_drop
                    and evaluation["fall_rate"]
                    <= baseline["fall_rate"] + cfg.max_fall_increase
                )
                record.update(evaluation=evaluation, accepted=accepted)
                # Only strictly successful training rollouts may enter replay.
                # Validation episodes are never used for fitting.
                if cfg.online_episodes:
                    seeds = []
                    while len(seeds) < cfg.online_episodes:
                        if (
                            split_for(online_seed, replay.manifest["checkpoint_sha256"])
                            == "train"
                        ):
                            seeds.append(online_seed)
                        online_seed += 1
                    episodes = rollouts.run(algorithm.actor, seeds)
                    record["online_successes_added"] = 0
                    record["online_unsuccessful_discarded"] = 0
                    for arrays, meta in episodes:
                        if not replay.add(arrays, meta):
                            record["online_unsuccessful_discarded"] += 1
                            continue
                        record["online_successes_added"] += 1
                        import numpy as np

                        archive = output / "online"
                        archive.mkdir(exist_ok=True)
                        if str(archive) not in replay.online_archives:
                            replay.online_archives.append(str(archive))
                        np.savez_compressed(archive / f"{meta['seed']}.npz", **arrays)
                        (archive / f"{meta['seed']}.json").write_text(
                            json.dumps(meta, indent=2) + "\n"
                        )
                if accepted:
                    baseline = evaluation
                    trust.accept()
                else:
                    trust.reject()
            elif ready:
                record["accepted"] = False
                trust.reject()
            record.update(
                accepted_policy_updates=trust.accepted_updates,
                consecutive_rejections=trust.consecutive_rejections,
                actor_learning_rate=algorithm.actor_optimizer.param_groups[0]["lr"],
                training_replay_episodes=len(replay.episodes),
                training_unsuccessful_episodes=sum(
                    not replay.is_success(e["meta"]) for e in replay.episodes
                ),
            )
            if rejection_limit_reached(cfg, trust.consecutive_rejections):
                record["stop_reason"] = "consecutive_candidate_regressions"
            if (
                evaluate_candidate
                or iteration % cfg.save_every == 0
                or iteration == cfg.iterations
                or record.get("stop_reason")
            ):
                save(
                    output / "last.pt",
                    algorithm,
                    cfg,
                    iteration,
                    baseline,
                    replay,
                    validation,
                    online_seed,
                    trust,
                )
                if record.get("accepted", False):
                    save(
                        output / "best.pt",
                        algorithm,
                        cfg,
                        iteration,
                        baseline,
                        replay,
                        validation,
                        online_seed,
                        trust,
                    )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            completed_iteration = iteration
            status = {
                "status": "running",
                "iteration": iteration,
                "stage": record["stage"],
                "accepted_policy_updates": trust.accepted_updates,
                "consecutive_rejections": trust.consecutive_rejections,
            }
            if record.get("stop_reason"):
                status.update(
                    status="stopped_no_progress", reason=record["stop_reason"]
                )
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=iteration)
            if record.get("stop_reason"):
                break
        trust.restore(trust.retained)
        save(
            output / "last.pt",
            algorithm,
            cfg,
            completed_iteration,
            baseline,
            replay,
            validation,
            online_seed,
            trust,
        )
        if not rejection_limit_reached(cfg, trust.consecutive_rejections):
            (output / "status.json").write_text(
                json.dumps(
                    {
                        "status": "stopped" if stopped else "completed",
                        "iteration": completed_iteration,
                        "accepted_policy_updates": trust.accepted_updates,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if tracker:
            tracker.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "courses/football-kicking/configs/soccer.json",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration/data/model wiring without optimization",
    )
    args = parser.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    if args.check:
        from .verify import check

        print(json.dumps(check(cfg), indent=2))
    else:
        train(cfg, args.resume)


if __name__ == "__main__":
    main()
