"""Headless pole training orchestrated through the vendored SuperMimic core."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import yaml

from . import ROOT
from .data import (
    CourseCase,
    PoleMotions,
    load_course,
    load_retention,
    write_vendor_motion,
)
from .environment import PoleEnvironment
from .models import (
    PoleActor,
    PoleCritic,
    PoleRobot,
    PoleSuperMimic,
    PoleWorldModel,
    observation_from_state,
    pole_actor_goal_features,
    pole_goal_features,
)
from .preflight import preflight
from .replay import PoleReplay
from .scene import build_course_scene


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_resume_best_checkpoint(resume_checkpoint: Path, run_dir: Path) -> Path | None:
    """Carry the best evaluated policy into a resumed run's output directory."""
    previous_best = resume_checkpoint.resolve().parent / "best.pt"
    if not previous_best.is_file():
        return None
    target = run_dir / "best.pt"
    shutil.copy2(previous_best, target)
    return target


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def case_from_retention(name: str) -> CourseCase:
    qpos = load_retention(name)
    path = ROOT / "pole_task/data/references" / f"{name}.npz"
    return CourseCase(
        name,
        "retention",
        np.array([[100, 0], [102, 0], [104, 0]], dtype=np.float32),
        qpos,
        path,
        digest(path),
    )


def select_training_cases(cases: list[CourseCase], config: dict) -> list[CourseCase]:
    by_id = {case.case_id: case for case in cases}
    case_ids = config.get("training_case_ids", ["nominal-01"])
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown pole training cases: {missing}")
    selected = [by_id[case_id] for case_id in case_ids]
    if any(case.split == "held_out" for case in selected):
        raise ValueError("Held-out pole cases cannot enter training data")
    return selected


def select_training_cases(cases: list[CourseCase], config: dict) -> list[CourseCase]:
    by_id = {case.case_id: case for case in cases}
    case_ids = config.get("training_case_ids", ["nominal-01"])
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown pole training cases: {missing}")
    selected = [by_id[case_id] for case_id in case_ids]
    if any(case.split == "held_out" for case in selected):
        raise ValueError("Held-out pole cases cannot enter training data")
    return selected


def observation_history(observations: np.ndarray, frame: int) -> np.ndarray:
    indexes = frame - np.arange(50)
    return observations[np.maximum(indexes, 0)].astype(np.float32, copy=True)


@torch.no_grad()
def run_episode(
    actor: PoleActor,
    case: CourseCase,
    scene_path: Path,
    *,
    seed: int,
    device: str,
    exploration: float = 0.0,
    show_gui: bool = False,
    goal_provider=None,
):
    env = PoleEnvironment(case, scene_path, seed=seed)
    observation = env.reset()
    observations, states = [observation.copy()], [env.state()]
    actions, rewards, terminals = [], [], []
    done = False
    rng = np.random.default_rng(seed)
    if show_gui:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    else:
        viewer = None
    try:
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            history_tensor = torch.as_tensor(env.history.copy(), dtype=torch.float32, device=device).unsqueeze(0)
            if goal_provider is None:
                raise ValueError("Pole playback requires the trained SuperMimic goal path")
            state_tensor = torch.as_tensor(env.state(), dtype=torch.float32, device=device).unsqueeze(0)
            goal = goal_provider(case, env.frame, state_tensor)
            action = actor.forward_control(obs_tensor, history_tensor, goal).squeeze(0).cpu().numpy()
            if exploration:
                action = np.clip(action + rng.normal(0, exploration, 22), -1, 1)
            next_observation, reward, done, transition = env.step(action)
            actions.append(transition["action"])
            rewards.append(reward)
            terminals.append(float(transition["terminal_cause"] in ("fall", "pole_collision")))
            observations.append(next_observation.copy())
            states.append(env.state())
            observation = next_observation
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
    finally:
        if viewer is not None:
            viewer.close()
    episode = {
        "case": case,
        "states": np.asarray(states, dtype=np.float32),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
        "result": env.result(),
        "transitions": env.transitions.copy(),
    }
    episode["returns"] = episode_returns(episode)
    return episode


def episode_returns(episode: dict, gamma: float = 0.99) -> np.ndarray:
    returns = np.zeros_like(episode["rewards"])
    future = 0.0
    for index in range(len(returns) - 1, -1, -1):
        future = float(episode["rewards"][index]) + gamma * future
        returns[index] = future
    return returns


def make_goal_provider(robot: PoleRobot, motions: PoleMotions, cases, device: str):
    clip_by_case = {case.case_id: index for index, case in enumerate(cases)}
    offsets = torch.arange(1, 11, dtype=torch.long, device=device)
    def goal_provider(case: CourseCase, frame: int, state: torch.Tensor):
        if not 0 <= frame < len(case.qpos):
            raise IndexError(f"Reference frame {frame} is outside {case.case_id}")
        clip = torch.tensor([clip_by_case[case.case_id]], dtype=torch.long, device=device)
        frame_tensor = torch.tensor([frame], dtype=torch.long, device=device)
        references = motions.sequence(clip, frame_tensor, offsets)
        return pole_actor_goal_features(robot, references, state)

    return goal_provider


def standalone_goal_provider(cases, scene_path: Path, device: str):
    scene = build_course_scene(cases[0], scene_path)
    robot = PoleRobot(scene).to(device)
    motion_dir = scene_path.parent / f"{scene_path.stem}-references"
    paths = [write_vendor_motion(case, motion_dir / f"{case.case_id}.npz") for case in cases]
    motions = PoleMotions(robot, list(cases), paths, device=device)
    motions.prepare()
    return make_goal_provider(robot, motions, cases, device)


def collect_episodes(actor, case, scene, count, *, seed, device, exploration, goal_provider):
    return [
        run_episode(
            actor, case, scene, seed=seed + index, device=device,
            exploration=exploration, goal_provider=goal_provider,
        )
        for index in range(count)
    ]


def make_replay(episodes, robot: PoleRobot, clip_by_case: dict[str, int], device: str):
    capacity = max(256, sum(len(row["actions"]) for row in episodes))
    replay = PoleReplay(
        capacity=capacity,
        num_envs=1,
        state_dim=79,
        context_dim=robot.context_dim,
        obs_dim=79,
        device=device,
    )
    for episode in episodes:
        case = episode["case"]
        clip = clip_by_case[case.case_id]
        poles = torch.as_tensor(case.poles_xy, dtype=torch.float32, device=device).unsqueeze(0)
        context = robot.context_for_poles(poles)
        for frame, transition in enumerate(episode["transitions"]):
            state = torch.as_tensor(transition["state"], dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.as_tensor(transition["next_state"], dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.as_tensor(transition["action"], dtype=torch.float32, device=device).unsqueeze(0)
            observation = torch.as_tensor(transition["observation"], dtype=torch.float32, device=device).unsqueeze(0)
            terminal = torch.tensor([episode["terminals"][frame]], dtype=torch.float32, device=device)
            boundary = torch.tensor([transition["terminal_cause"] is not None], dtype=torch.bool, device=device)
            replay.add(
                state,
                next_state,
                action,
                context,
                observation,
                terminal,
                boundary,
                torch.tensor([clip], dtype=torch.long, device=device),
                torch.tensor([frame], dtype=torch.long, device=device),
            )
    return replay


def sample_model_metrics(algorithm: PoleSuperMimic, replay: PoleReplay, *, count: int, batch_size: int):
    robot, motions, model = algorithm.robot, algorithm.motions, algorithm.model
    device = replay.states.device
    transition_errors, term_losses = [], []
    remaining = count
    while remaining:
        size = min(batch_size, remaining)
        batch = replay.sample(size, window=1, history=1)
        state, action = batch.states[:, 0], batch.actions[:, 0]
        predicted = model(state, action, batch.context)
        target = batch.next_states[:, 0]
        target_reference = motions.sequence(batch.clips, batch.frames, torch.tensor([1], device=device))[:, 0]
        logits = model.termination_logits(robot, predicted, target_reference)
        scale = model.scale[:57]
        transition_errors.append(((predicted[:, :57] - target[:, :57]) / scale).square().mean(-1))
        term_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.terminated[:, 0], reduction="none"))
        remaining -= size
    return {
        "held_out_transition_rmse": float(torch.cat(transition_errors).mean().sqrt()),
        "held_out_termination_bce": float(torch.cat(term_losses).mean()),
    }


def sample_real_returns(episodes, cases, motions: PoleMotions, robot: PoleRobot, batch_size: int, device: str):
    choices = np.random.randint(0, len(episodes), size=batch_size)
    rows = [episodes[int(choice)] for choice in choices]
    frames = np.array([np.random.randint(len(row["actions"])) for row in rows])
    clips = torch.tensor([next(i for i, case in enumerate(cases) if case.case_id == row["case"].case_id) for row in rows], device=device)
    frame_tensor = torch.as_tensor(frames, dtype=torch.long, device=device)
    states = torch.as_tensor(np.stack([row["states"][frame] for row, frame in zip(rows, frames, strict=True)]), device=device)
    targets = torch.as_tensor(np.array([row["returns"][frame] for row, frame in zip(rows, frames, strict=True)]), device=device)
    references = motions.sequence(clips, frame_tensor, torch.arange(10, device=device))
    poles = references[:, 0, 79:85].reshape(-1, 3, 2)
    goal = pole_goal_features(robot, references, states)
    context = robot.context_for_poles(poles)
    return states, goal, context, targets


def critic_relative_rmse(algorithm: PoleSuperMimic, episodes, cases, motions, robot, *, batch_size: int, device: str):
    states, goals, contexts, targets = sample_real_returns(episodes, cases, motions, robot, batch_size, device)
    with torch.no_grad():
        values = algorithm.critic(states, goals, contexts)
        return float((values - targets).square().mean().sqrt() / targets.square().mean().sqrt().clamp_min(1e-6))


def critic_and_model_gates(model_metrics, critic_metrics, config):
    return (
        model_metrics["held_out_transition_rmse"] <= config["held_out_model_rmse_max"]
        and model_metrics["held_out_termination_bce"] <= config["held_out_termination_bce_max"]
        and critic_metrics["critic_relative_rmse"] <= config["critic_relative_rmse_max"]
    )


def guarded_actor_update(algorithm, batch, gates):
    """Keep the vendor actor optimizer behind the measured pretraining gates."""
    if not gates.get("actor_updates_allowed", False):
        return None
    return algorithm.improve_policy(batch)


def checkpoint_payload(actor, world, critic, algorithm, config, identity, *, update, stage, gates=None, evaluation=None):
    return {
        "format": "q1_pole_weaving_project_v9",
        "stage": stage,
        "actor": actor.state_dict(),
        "world_model": world.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": algorithm.actor_optimizer.state_dict(),
        "model_optimizer": algorithm.model_optimizer.state_dict(),
        "critic_optimizer": algorithm.critic_optimizer.state_dict(),
        "config": config,
        "identity": identity,
        "gates": gates,
        "evaluation": evaluation,
        "update": update,
    }


def evaluate(actor, cases, scene_path, *, device: str, seed: int, goal_provider=None):
    if goal_provider is None:
        goal_provider = standalone_goal_provider(
            cases, scene_path.with_name(f"{scene_path.stem}-goals.xml"), device
        )
    reports = {}
    for index, case in enumerate(cases):
        reports[case.case_id] = run_episode(
            actor, case, scene_path, seed=seed + index, device=device,
            goal_provider=goal_provider,
        )["result"]
    return reports


def gate_evaluation(reports, config):
    for name in ("nominal-01", "eval-01"):
        row = reports[name]
        if (
            not row["route_completed"]
            or row["pole_collision"]
            or row["fall"]
            or row["root_xy_rmse_m"] > config["course_success_rmse_m"]
        ):
            return False
    for name in config["retention_case_ids"]:
        row = reports[name]
        if (
            row["fall"]
            or row["root_xy_rmse_m"] > config["retention_rmse_m"]
        ):
            return False
    return True


def retention_candidate_is_safe(reports, baseline, config):
    return all(
        not reports[name]["fall"]
        and reports[name]["root_xy_rmse_m"] <= baseline[name]["root_xy_rmse_m"]
        for name in config["retention_case_ids"]
    )


def course_candidate_is_safe(reports, baseline, config):
    tolerance = float(config.get("course_regression_tolerance_m", 0.0))
    for name in ("nominal-01", "eval-01"):
        candidate, anchor = reports[name], baseline[name]
        if anchor.get("fall") or anchor.get("pole_collision"):
            if candidate["steps"] < anchor["steps"]:
                return False
            if candidate.get("fall", False) and not anchor.get("fall"):
                return False
            if candidate.get("pole_collision", False) and not anchor.get("pole_collision"):
                return False
            continue
        if candidate.get("fall", False) or candidate.get("pole_collision", False):
            return False
        if anchor.get("route_completed") and not candidate.get("route_completed"):
            return False
        if candidate["root_xy_rmse_m"] > anchor["root_xy_rmse_m"] + tolerance:
            return False
    return True


def _episode_imagined_batch(episodes, cases, motions, robot, batch_size, device):
    states, goals, contexts, returns = sample_real_returns(
        episodes, cases, motions, robot, batch_size, device
    )
    return states, goals, contexts, returns


def train(config_path: Path, resume_checkpoint: Path | None = None) -> Path:
    config_path = config_path.resolve()
    preflight_report = preflight(config_path)
    if not preflight_report["initialization"]["available"]:
        raise FileNotFoundError(preflight_report["training_blocker"])
    if not preflight_report["training_ready"]:
        raise ValueError(preflight_report["training_blocker"])
    if not preflight_report["hardware"]["ready"]:
        raise RuntimeError(
            "Requested training device failed preflight: "
            + json.dumps(preflight_report["hardware"], sort_keys=True)
        )
    config = json.loads(config_path.read_text())
    root = config_path.parents[1]
    expert_path = (root / config["expert_checkpoint"]).resolve()
    manifest_path = (root / config["manifest"]).resolve()
    output_root = (root / config["output_root"]).resolve()
    seed, device = int(config["seed"]), config["device"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; set device=cpu explicitly")
        torch.cuda.set_device(device)
        free_bytes, _ = torch.cuda.mem_get_info()
        minimum = float(config["gpu_min_free_gb"]) * 1024**3
        if free_bytes < minimum:
            raise RuntimeError(
                f"Insufficient free GPU memory: {free_bytes / 1024**3:.2f} GiB; "
                f"need {minimum / 1024**3:.2f} GiB"
            )
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    nominal = load_course("nominal-01", manifest_path=manifest_path)
    heldout = load_course("eval-01", manifest_path=manifest_path)
    retention = [case_from_retention(name) for name in config["retention_case_ids"]]
    cases = [nominal, heldout, *retention]
    max_evaluation_steps = nominal.control_steps + heldout.control_steps + sum(
        len(row.qpos) - 1 for row in retention
    )
    training_cases = select_training_cases(cases, config)
    maximum_evaluations = int(config["max_actor_updates"]) // int(config["evaluation_interval"])
    worst_case_steps = (
        sum(
            (int(config["collection_episodes"]) if row.case_id == nominal.case_id else int(config["retention_collection_episodes"]))
            * row.control_steps
            for row in training_cases
        )
        + int(config["held_out_validation_episodes"]) * heldout.control_steps
        + nominal.control_steps + heldout.control_steps
        + sum(len(row.qpos) - 1 for row in retention)
        + maximum_evaluations * max_evaluation_steps
    )
    if worst_case_steps > int(config["max_environment_steps"]):
        raise ValueError(
            f"Configured environment-step budget {config['max_environment_steps']} "
            f"cannot cover the bounded collection/evaluation plan ({worst_case_steps})"
        )
    expert_cfg = yaml.safe_load((root / "expert/configs/Q1.yaml").read_text())
    actor, mapping = PoleActor.from_project_expert(expert_path, expert_cfg)
    actor = actor.to(device)
    baseline = deepcopy(actor).to(device).eval()
    resume_state = None
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        if resume_state.get("format") != "q1_pole_weaving_project_v9":
            raise ValueError("Resume checkpoint is not a project-trained SuperMimic pole checkpoint")
        if resume_state["identity"].get("initialization_sha256") != digest(expert_path):
            raise ValueError("Resume checkpoint was trained from a different Q1 expert initialization")
        if resume_state.get("config") != config:
            raise ValueError("Resume checkpoint configuration differs from the requested pole run")
        actor.load_state_dict(resume_state["actor"], strict=True)
    run_dir = output_root / time.strftime("run-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    resumed_best = restore_resume_best_checkpoint(resume_checkpoint, run_dir) if resume_checkpoint else None
    motion_dir = run_dir / "references"
    vendor_paths = [write_vendor_motion(case, motion_dir / f"{case.case_id}.npz") for case in cases]
    nominal_scene = run_dir / "nominal.xml"
    PoleEnvironment(nominal, nominal_scene)
    robot = PoleRobot(nominal_scene).to(device)
    motions = PoleMotions(robot, cases, vendor_paths, device=device)
    motions.prepare()
    clip_by_case = {case.case_id: index for index, case in enumerate(cases)}
    goal_provider = make_goal_provider(robot, motions, cases, device)

    world = PoleWorldModel(robot, tuple(config["model_hidden"])).to(device)
    critic = PoleCritic(robot).to(device)
    algorithm = PoleSuperMimic(
        robot, motions, world, actor, critic,
        learning_rate=float(config["actor_learning_rate"]),
        horizon=int(config["horizon"]),
    )
    for group in algorithm.model_optimizer.param_groups:
        group["lr"] = float(config.get("model_learning_rate", 3e-4))
    for group in algorithm.critic_optimizer.param_groups:
        group["lr"] = float(config.get("critic_learning_rate", 3e-4))
    if resume_state is not None:
        world.load_state_dict(resume_state["world_model"], strict=True)
        critic.load_state_dict(resume_state["critic"], strict=True)
        algorithm.actor_optimizer.load_state_dict(resume_state["actor_optimizer"])
        algorithm.model_optimizer.load_state_dict(resume_state["model_optimizer"])
        algorithm.critic_optimizer.load_state_dict(resume_state["critic_optimizer"])
    identity = {
        "format": "q1_pole_weaving_project_v9",
        "seed": seed,
        "initialization_sha256": digest(expert_path),
        "initialization_path": str(expert_path),
        "source_task_actor_loaded": False,
        "actor_weight_mapping": mapping,
        "data_manifest_sha256": digest(manifest_path),
        "data_split": {
            "train": {row.case_id: row.identity for row in training_cases},
            "held_out": [heldout.identity],
            "retention": {row.case_id: row.sha256 for row in retention},
        },
        "budgets": {
            key: config[key]
            for key in ("max_environment_steps", "max_actor_updates", "evaluation_interval", "success_patience")
        },
        "robot_scene_sha256": digest(root / "expert/resources/q1_description/q1_soccer.xml"),
        "vendor_commit": "118338b19dcd14f768d1e7de5a48d04b78987688",
        "started_unix": time.time(),
        "resume_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
        "resume_checkpoint_sha256": digest(resume_checkpoint) if resume_checkpoint else None,
        "resumed_best_checkpoint": str(resumed_best) if resumed_best else None,
    }
    atomic_json(run_dir / "identity.json", identity)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    validation = collect_episodes(
        baseline, heldout, run_dir / "heldout.xml", int(config["held_out_validation_episodes"]),
        seed=seed + 10000, device=device, exploration=0.02, goal_provider=goal_provider,
    )
    training = []
    for index, case in enumerate(training_cases):
        scene_path = nominal_scene if case.case_id == nominal.case_id else run_dir / f"collection-{case.case_id}.xml"
        episode_count = int(
            config["collection_episodes"]
            if case.case_id == nominal.case_id
            else config["retention_collection_episodes"]
        )
        training.extend(
            collect_episodes(
                actor, case, scene_path, episode_count,
                seed=seed + 1000 + index * 100000,
                device=device, exploration=float(config["exploration_std"]),
                goal_provider=goal_provider,
            )
        )
    env_steps = sum(len(row["actions"]) for row in training + validation)
    if env_steps > int(config["max_environment_steps"]):
        stop = {"reason": "max_environment_steps_during_collection", "environment_steps": env_steps}
        atomic_json(run_dir / "stop_reason.json", stop)
        identity.update({"best_checkpoint": None, "stop_reason": stop, "environment_steps": env_steps})
        atomic_json(run_dir / "identity.json", identity)
        return run_dir
    train_replay = make_replay(training, robot, clip_by_case, device)
    heldout_replay = make_replay(validation, robot, clip_by_case, device)
    (run_dir / "replay").mkdir()
    for split, episodes in (("train", training), ("heldout", validation)):
        for index, episode in enumerate(episodes):
            np.savez_compressed(
                run_dir / "replay" / f"{split}-{index:04d}.npz",
                states=episode["states"], observations=episode["observations"],
                actions=episode["actions"], rewards=episode["rewards"],
                terminals=episode["terminals"], reference_id=np.full(len(episode["actions"]), episode["case"].identity),
                pole_xy_m=np.repeat(episode["case"].poles_xy[None], len(episode["actions"]), axis=0),
                pole_contacts=np.asarray([json.dumps(row["pole_contacts"]) for row in episode["transitions"]]),
                terminal_causes=np.asarray([row["terminal_cause"] or "none" for row in episode["transitions"]]),
            )
    atomic_json(run_dir / "identity.json", {**identity, "environment_steps": env_steps})

    if resume_state is not None and resume_state.get("stage") == "actor":
        gates = dict(resume_state["gates"])
    else:
        for _ in range(int(config["model_pretrain_updates"])):
            batch = train_replay.sample(int(config["batch_size"]), window=8, history=1)
            algorithm.fit_model(batch)
        model_metrics = sample_model_metrics(
            algorithm, heldout_replay, count=int(config["held_out_validation_episodes"]) * 128,
            batch_size=int(config["batch_size"]),
        )
        for _ in range(int(config["critic_warmup_updates"])):
            states, goals, contexts, returns = _episode_imagined_batch(
                training, cases, motions, robot, int(config["batch_size"]), device
            )
            algorithm._fit_critic(
                states, goals, contexts, returns, torch.ones_like(returns), 1, 1
            )
        critic_metrics = {
            "critic_relative_rmse": critic_relative_rmse(
                algorithm, validation, cases, motions, robot,
                batch_size=min(2048, int(config["batch_size"]) * 8), device=device,
            )
        }
        gates = {
            **model_metrics,
            **critic_metrics,
            "actor_updates_allowed": critic_and_model_gates(model_metrics, critic_metrics, config),
        }
    atomic_json(run_dir / "pretraining-gates.json", gates)
    if not gates["actor_updates_allowed"]:
        stop = {"reason": "world_model_or_critic_gate_failed", **gates}
        atomic_json(run_dir / "stop_reason.json", stop)
        identity.update({"best_checkpoint": None, "stop_reason": stop, "environment_steps": env_steps})
        atomic_json(run_dir / "identity.json", identity)
        return run_dir

    course_baseline = evaluate(
        baseline, [nominal, heldout], run_dir / "course-baseline.xml",
        device=device, seed=seed + 20000, goal_provider=goal_provider,
    )
    retention_baseline = evaluate(
        baseline, retention, run_dir / "retention.xml", device=device,
        seed=seed + 20500, goal_provider=goal_provider,
    )
    env_steps += sum(row["steps"] for row in course_baseline.values())
    env_steps += sum(row["steps"] for row in retention_baseline.values())
    identity["course_baseline"] = course_baseline
    identity["retention_baseline"] = retention_baseline
    identity["retention_baseline"] = retention_baseline
    identity["environment_steps"] = env_steps
    atomic_json(run_dir / "identity.json", identity)
    if env_steps >= int(config["max_environment_steps"]):
        stop = {"reason": "max_environment_steps", "environment_steps": env_steps}
        atomic_json(run_dir / "stop_reason.json", stop)
        identity.update({"best_checkpoint": None, "stop_reason": stop})
        atomic_json(run_dir / "identity.json", identity)
        return run_dir
    best_score = float(resume_state.get("best_score", float("inf"))) if resume_state else float("inf")
    best_path, patience = run_dir / "best.pt", int(resume_state.get("patience", 0)) if resume_state else 0
    accepted_state = deepcopy(actor.state_dict())
    accepted_actor = deepcopy(actor).eval()
    start_update = int(resume_state.get("update", 0)) + 1 if resume_state else 1
    if resume_state is None or resume_state.get("stage") != "actor":
        torch.save(
            checkpoint_payload(actor, world, critic, algorithm, config, identity, update=0, stage="actor", gates=gates),
            run_dir / "latest.pt",
        )
    for update in range(start_update, int(config["max_actor_updates"]) + 1):
        batch = train_replay.sample(int(config["batch_size"]), window=1, history=50)
        metrics = guarded_actor_update(algorithm, batch, gates)
        if metrics is None:
            atomic_json(run_dir / "stop_reason.json", {"reason": "actor_gate_closed", "update": update})
            break
        references = motions.sequence(
            batch.clips, batch.frames, torch.arange(1, 11, device=device)
        )
        goals = pole_actor_goal_features(robot, references, batch.states[:, 0])
        with torch.no_grad():
            observation = batch.history[:, 0]
            anchor = accepted_actor.forward_control(observation, batch.history, goals)
            candidate = actor.forward_control(observation, batch.history, goals)
            delta = candidate - anchor
            action_rms = float(delta.square().mean().sqrt())
            action_max = float(delta.abs().max())
        metrics.update({"action_delta_rms": action_rms, "action_delta_max": action_max})
        if action_rms > config["trust_action_rms"] or action_max > config["trust_action_max"]:
            actor.load_state_dict(accepted_state)
            atomic_json(run_dir / "stop_reason.json", {"reason": "actor_action_trust_limit", "update": update, "metrics": metrics})
            break
        if update % int(config["evaluation_interval"]):
            continue
        if env_steps + max_evaluation_steps > int(config["max_environment_steps"]):
            stop = {"reason": "max_environment_steps_before_evaluation", "update": update, "environment_steps": env_steps}
            atomic_json(run_dir / "stop_reason.json", stop)
            break
        reports = evaluate(
            actor, [nominal, heldout, *retention], nominal_scene,
            device=device, seed=seed + 30000, goal_provider=goal_provider,
        )
        env_steps += sum(row["steps"] for row in reports.values())
        course_ok = course_candidate_is_safe(reports, course_baseline, config)
        if not course_ok:
            actor.load_state_dict(accepted_state)
            atomic_json(run_dir / "stop_reason.json", {"reason": "course_trust_gate_failed", "update": update, "metrics": metrics, "reports": reports, "baseline": course_baseline})
            break
        retention_ok = retention_candidate_is_safe(reports, retention_baseline, config)
        if not retention_ok:
            actor.load_state_dict(accepted_state)
            atomic_json(run_dir / "stop_reason.json", {"reason": "retention_trust_gate_failed", "update": update, "metrics": metrics, "reports": reports})
            break
        accepted_state = deepcopy(actor.state_dict())
        accepted_actor = deepcopy(actor).eval()
        score = reports["nominal-01"]["root_xy_rmse_m"] + reports["eval-01"]["root_xy_rmse_m"]
        latest = checkpoint_payload(
            actor, world, critic, algorithm, config, identity,
            update=update, stage="actor", gates=gates, evaluation=reports,
        )
        is_best = gate_evaluation(reports, config) and score < best_score
        if is_best:
            best_score, patience = score, 0
        else:
            patience += 1
        latest.update({"best_score": best_score, "patience": patience})
        torch.save(latest, run_dir / "latest.pt")
        if is_best:
            torch.save(latest, best_path)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"update": update, "environment_steps": env_steps, "metrics": metrics, "evaluation": reports, "best_score": best_score, "patience": patience}) + "\n")
        if env_steps >= int(config["max_environment_steps"]):
            atomic_json(run_dir / "stop_reason.json", {"reason": "max_environment_steps", "update": update, "environment_steps": env_steps})
            break
        if patience >= int(config["success_patience"]):
            atomic_json(run_dir / "stop_reason.json", {"reason": "success_patience_exhausted", "update": update, "best_score": best_score})
            break
    stop_reason = json.loads((run_dir / "stop_reason.json").read_text()) if (run_dir / "stop_reason.json").exists() else "budget_exhausted"
    identity.update({
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "stop_reason": stop_reason,
        "environment_steps": env_steps,
    })
    atomic_json(run_dir / "identity.json", identity)
    return run_dir


def main(argv=None):
    # The pole-weaving profile trains the source-compatible 78/149 contract and
    # starts strictly from the registered task 7.7 warmup checkpoint.  The
    # legacy project-expert helpers above remain importable for their tests,
    # but the entrypoint never dispatches to that checkpoint lineage.
    from .source_training import main as source_main

    return source_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
