"""Regenerate source task 7.7 observation/goal/actor fixtures.

Run from the registered source project so that ``import supermimic`` resolves
to the source implementation, not this project's vendored pin::

    cd /path/to/SuperMimic
    uv run python /path/to/Q1-Robot-Soccer/courses/pole-weaving/pole_task/tools/generate_source_fixtures.py \
        --checkpoint runs/q1-pole-weaving/stage-7-7/course-warmup-v10/checkpoint_2000.pt \
        --reference /path/to/Q1-Robot-Soccer/courses/pole-weaving/pole_task/data/references/nominal-01.npz \
        --robot-xml assets/q1_v3/xml/q1_v3_mc.xml \
        --output /path/to/Q1-Robot-Soccer/courses/pole-weaving/pole_task/data/fixtures/source-policy-fixture.npz

The output is a deterministic input/output sample of the source robot
observation, robot-relative goal encoding and actor forward pass.  Tests compare
this project's profile against it without importing the source project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Source project root used to verify the imported supermimic package",
    )
    parser.add_argument("--frame", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    import supermimic
    from supermimic.data import Q1Motions
    from supermimic.models import MotionPolicy, goal_features, load_actor_state
    from supermimic.randomization import nominal_context
    from supermimic.robot import Q1Robot

    source_module = Path(supermimic.__file__).resolve()
    source_root = args.source_root.resolve()
    if source_root != source_module and source_root not in source_module.parents:
        raise SystemExit(
            f"Imported supermimic from {source_module}, not the source root {source_root}; "
            "run this generator from the source project"
        )

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    robot = Q1Robot(
        args.robot_xml.resolve(),
        reward_scaling="q1_proportional" if config.get("q1_reward_scaling") else None,
        modeled_action_delay=config.get("modeled_action_delay", False),
        course_conditioned=True,
        goal_representation=config.get("goal_representation", "absolute"),
    )
    robot.set_course_frame((0.0, 0.0, 0.0), 0.0, 0.04)

    with np.load(args.reference.resolve(), allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32).copy()
        poles = np.asarray(data["synthesis_layout_xy_m"], dtype=np.float32).copy()
    frame = args.frame
    if not 0 < frame < len(qpos) - 11:
        raise SystemExit("Fixture frame must leave room for history and future windows")
    motions = Q1Motions(robot, [args.reference.resolve()], dt=0.02, device="cpu", storage="device")
    motions.prepare()
    clip = torch.tensor([0])
    references = motions.sequence(clip, torch.tensor([frame]), torch.arange(1, 11))

    def state_at(index: int) -> torch.Tensor:
        qvel = 0.05 * np.sin(np.arange(28, dtype=np.float32) + 0.1 * index)
        action = 0.2 * np.cos(np.arange(22, dtype=np.float32) - 0.1 * index)
        return torch.from_numpy(np.concatenate((qpos[index], qvel, action))).unsqueeze(0)

    context = nominal_context(robot.context_layout, 1)
    context[:, robot.context_layout.motor_offset] = torch.linspace(-0.01, 0.01, 22)
    context[:, robot.context_layout.delay_s] = 0.004
    context[:, robot.context_layout.course_poles] = torch.from_numpy(
        poles.reshape(1, 6)
    )
    history = torch.cat(
        [robot.observation(state_at(index), context) for index in range(frame - 9, frame + 1)],
        dim=0,
    )
    state = state_at(frame)
    goal = goal_features(robot, references, state)
    actor = MotionPolicy(robot, (2048, 1024, 512, 256), activation="silu")
    load_actor_state(actor, checkpoint["actor"])
    with torch.no_grad():
        action = actor(history.unsqueeze(0), goal)
        target = robot.default_q + context[:, robot.context_layout.motor_offset] + (
            config["action_scale"] * action
        )

    arrays = {
        "state": state[0],
        "context": context[0],
        "history": history,
        "observation": history[-1],
        "references": references[0],
        "goal": goal[0],
        "action": action[0],
        "pd_target": target[0],
        "previous_action": history[-1, 50:72],
        "noise_scale": robot.noise_scale(),
        "default_q": robot.default_q,
        "poles_xy": torch.from_numpy(poles),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{name: value.cpu().numpy() for name, value in arrays.items()})
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = checkpoint.get("identity", {}).get("source_revision")
    metadata = {
        "format": "supermimic_source_policy_fixture_v1",
        "source_revision": revision,
        "source_checkout": str(source_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference.resolve()),
        "robot_xml": str(args.robot_xml.resolve()),
        "robot_xml_sha256": sha256(args.robot_xml.resolve()),
        "frame": frame,
        "goal_representation": config.get("goal_representation"),
        "action_scale": config["action_scale"],
        "history_frames": 10,
        "observation_dim": int(history.shape[-1]),
        "context_dim": int(context.shape[-1]),
        "goal_dim": int(goal.shape[-1]),
        "action_dim": int(action.shape[-1]),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
