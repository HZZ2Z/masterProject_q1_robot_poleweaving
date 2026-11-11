"""Wiring checks and explicitly bounded, disposable optimizer smoke tests."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import ROOT
from .data import ExpertReplay
from .robot import physical_states
from .sensors import native_sensors
from .vendor import identity
from .train import TrainConfig, build, reserve_device
from .rollout import NativeRollouts
from expert.control import BoundedSoccerPlayer
from expert.collect import collect_episode
from supermimic.algorithm import SuperMimic
from supermimic.robot import quat_apply


def sensor_parity(robot):
    with contextlib.redirect_stdout(io.StringIO()):
        player = BoundedSoccerPlayer(seed=42, reset_mode="default")
    arrays, meta = collect_episode(player, max_control_steps=12)
    states = physical_states(
        arrays, native_sensors(arrays, ROOT / "logs/sensor_smoke_cache")
    )
    maximum = 0.0
    camera_error = 0.0
    for t in range(meta["steps"]):
        state = states[t : t + 1].to(robot.neutral.device)
        player.data.qpos[:] = arrays["qpos"][t]
        player.data.qvel[:] = arrays["qvel"][t]
        mujoco.mj_forward(player.model, player.data)
        p, r, _, _ = robot.kinematics(state)
        camera = p[:, robot.camera_body] + quat_apply(
            r[:, robot.camera_body], robot.camera_position
        )
        camera_error = max(
            camera_error,
            float(
                np.abs(
                    camera.detach().cpu().numpy()[0]
                    - player.data.sensor("camera_position").data
                ).max()
            ),
        )
        queue = arrays["ball_observation_queue"][t]
        previous = np.concatenate((queue[1:], queue[-1:]), 0)
        context = torch.tensor(
            np.concatenate(
                (previous.reshape(-1), [meta["ball_delay_steps"], 1 - t % 2])
            )[None],
            device=state.device,
            dtype=torch.float32,
        )
        q = state[0, 95:99]
        w, x, y, z = q
        yaw = torch.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
        delta = state[0, 79:81] - state[0, 92:94]
        relative = torch.stack(
            (
                yaw.cos() * delta[0] + yaw.sin() * delta[1],
                -yaw.sin() * delta[0] + yaw.cos() * delta[1],
            )
        )
        visible = bool(queue[0, 2])
        normals = (torch.tensor(queue[0, :2], device=state.device) - relative) / (
            0.149 + 0.124 * relative.norm()
        )
        innovation = torch.cat((state.new_tensor([0.0 if visible else 1.0]), normals))[
            None
        ]
        observed, _ = robot.observe(state, context, innovation=innovation)
        difference = float(
            np.abs(observed.detach().cpu().numpy()[0] - arrays["observations"][t]).max()
        )
        maximum = max(maximum, difference)
    if maximum > 2e-5 or camera_error > 2e-6:
        raise AssertionError(
            f"Differentiable sensor/native mismatch: {maximum=}, {camera_error=}"
        )
    return dict(
        sensor_max_absolute_error=maximum,
        camera_fk_max_absolute_error=camera_error,
        frames=meta["steps"],
    )


def check(cfg, *, smoke=False):
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    resources = reserve_device(cfg)
    # Strict isolation: this function never creates the formal output directory.
    replay = ExpertReplay(cfg.dataset, seed=cfg.seed)
    validation = ExpertReplay(cfg.dataset, split="validation", seed=123)
    algorithm = build(cfg, replay)
    assert type(algorithm).fit_model_tensors is SuperMimic.fit_model_tensors
    assert type(algorithm)._update_actor is SuperMimic._update_actor
    assert type(algorithm)._fit_critic is SuperMimic._fit_critic
    train_seeds = {e["meta"]["seed"] for e in replay.episodes}
    validation_seeds = {e["meta"]["seed"] for e in validation.episodes}
    assert not train_seeds & validation_seeds
    batch = replay.sample(2, window=8, device=cfg.device)
    with torch.no_grad():
        predicted = algorithm.actor(batch.history)
        # Native teacher and initialized student use precisely the same inputs.
        from expert.policy import ActorCritic

        teacher = ActorCritic(algorithm.robot.cfg).to(cfg.device)
        teacher.load_state_dict(
            torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=True)[
                "model"
            ]
        )
        expected = teacher(batch.history[:, 0], batch.history)
        teacher_error = float((predicted - expected).abs().max())
        assert teacher_error == 0
        next_state = algorithm.model(
            batch.states[:, 0], batch.actions[:, 0], batch.context
        )
        assert next_state.shape == (2, 109) and torch.isfinite(next_state).all()
        assert torch.equal(next_state[:, 57:79], batch.actions[:, 0])
    report = dict(
        passed=True,
        device=cfg.device,
        hardware=resources,
        vendor_commit=identity()["commit"],
        vendor_core_unmodified=True,
        training_episodes=len(replay.episodes),
        successful_training_references=len(replay.successes),
        validation_episodes=len(validation.episodes),
        teacher_raw_action_max_error=teacher_error,
        control_contract=replay.manifest["control_contract"],
        formal_training_started=False,
        **sensor_parity(algorithm.robot),
    )
    if not report["vendor_core_unmodified"]:
        raise AssertionError("Manufacturer source has changed")
    if smoke:
        before = {
            key: value.detach().clone()
            for key, value in algorithm.model.named_parameters()
        }
        report["world_update"] = algorithm.fit_model(batch)
        model_delta = max(
            float((value - before[key]).abs().max())
            for key, value in algorithm.model.named_parameters()
        )
        policy_before = {
            key: value.detach().clone()
            for key, value in algorithm.actor.named_parameters()
        }
        policy_batch = replay.sample(2, window=1, imitation=True, device=cfg.device)
        report["policy_update"] = algorithm.improve_policy(
            policy_batch,
            critic_epochs=1,
            critic_minibatches=1,
            anchor_batch=policy_batch,
        )
        policy_delta = max(
            float((value - policy_before[key]).abs().max())
            for key, value in algorithm.actor.named_parameters()
        )
        if min(model_delta, policy_delta) <= 0:
            raise AssertionError("World model or actor did not update")
        report.update(
            world_parameter_max_delta=model_delta,
            actor_parameter_max_delta=policy_delta,
        )
        # Serialize and restore the actual task modules, then drive native physics
        # from the restored student. No smoke checkpoint can be used as best.pt.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="q1-smoke-") as folder:
            path = Path(folder) / "state.pt"
            torch.save(
                {
                    name: getattr(algorithm, name).state_dict()
                    for name in [
                        "actor",
                        "model",
                        "critic",
                        "actor_optimizer",
                        "model_optimizer",
                        "critic_optimizer",
                    ]
                },
                path,
            )
            saved = torch.load(path, map_location=cfg.device, weights_only=False)
            for name, state in saved.items():
                getattr(algorithm, name).load_state_dict(state)
        report["native_rollout"] = NativeRollouts(1).evaluate(
            algorithm.actor, [991001, 991002], max_steps=12
        )
        report["checkpoint_roundtrip"] = True
        if cfg.device.startswith("cuda"):
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "courses/football-kicking/configs/soccer.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "logs/vendor_wiring_check.json"
    )
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config)
    cfg.device = args.device
    if args.smoke:
        cfg.horizon = 3
        cfg.batch_size = 2
        cfg.gpu_memory_gb = 0.75
        cfg.gpu_reserve_gb = 4.0
    report = check(cfg, smoke=args.smoke)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
