import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pole_task import COURSE_ROOT, ROOT
from pole_task.source import verify_source_manifest
from pole_task.source_profile import (
    SOURCE_ACTION_DIM,
    SOURCE_ACTION_SCALE,
    SOURCE_CONTEXT_DIM,
    SOURCE_GOAL_DIM,
    SOURCE_HISTORY_FRAMES,
    SOURCE_OBS_DIM,
    SOURCE_STATE_DIM,
    PoleContextLayout,
    SourcePoleRobot,
    build_source_models,
    source_goal_features,
    validate_source_history,
)

MANIFEST_PATH = COURSE_ROOT / "pole_task/source_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _fixture() -> dict[str, np.ndarray]:
    manifest = _manifest()
    fixture = manifest["policy_fixture"]
    path = COURSE_ROOT / "pole_task" / fixture["project_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == fixture["sha256"]
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _robot() -> tuple[SourcePoleRobot, dict]:
    source = verify_source_manifest(MANIFEST_PATH)
    robot_xml = Path(source["artifact_paths"]["robot_xml"])
    robot = SourcePoleRobot(robot_xml)
    checkpoint = torch.load(
        Path(source["artifact_paths"][source["warmup_checkpoint_id"]]),
        map_location="cpu",
        weights_only=False,
    )
    robot.load_state_dict(checkpoint["robot"], strict=True)
    robot.set_course_frame((0.0, 0.0, 0.0), 0.0, 0.04)
    return robot, checkpoint


def test_source_profile_dimensions_and_context_layout_match_the_manifest():
    manifest = _manifest()
    fixture_metadata = json.loads(
        (COURSE_ROOT / "pole_task" / manifest["policy_fixture"]["metadata_path"]).read_text()
    )

    assert SOURCE_OBS_DIM == manifest["observation_contract"]["per_frame_dim"] == 78
    assert SOURCE_HISTORY_FRAMES == manifest["observation_contract"]["history_frames"] == 10
    assert SOURCE_STATE_DIM == manifest["tensor_shapes"]["model"]["scale"][0] == 79
    assert SOURCE_GOAL_DIM == manifest["goal_contract"]["dim"] == 570
    assert SOURCE_ACTION_DIM == manifest["action_contract"]["dim"] == 22
    assert SOURCE_ACTION_SCALE == manifest["action_contract"]["scale"] == 0.5
    assert SOURCE_CONTEXT_DIM == manifest["context_contract"]["dim"] == 149
    assert fixture_metadata["observation_dim"] == SOURCE_OBS_DIM
    assert fixture_metadata["context_dim"] == SOURCE_CONTEXT_DIM
    assert fixture_metadata["goal_dim"] == SOURCE_GOAL_DIM
    assert fixture_metadata["action_dim"] == SOURCE_ACTION_DIM

    layout = PoleContextLayout.build(22, 26)
    assert layout.size == SOURCE_CONTEXT_DIM
    for name, (start, stop) in manifest["context_contract"]["slices"].items():
        span = getattr(layout, name)
        assert [span.start, span.stop] == [start, stop]


def test_source_robot_observation_matches_the_source_fixture_exactly():
    fixture = _fixture()
    robot, _ = _robot()
    state = torch.from_numpy(fixture["state"]).unsqueeze(0)
    context = torch.from_numpy(fixture["context"]).unsqueeze(0)

    observation = robot.observation(state, context)

    assert observation.shape == (1, SOURCE_OBS_DIM)
    assert torch.equal(observation[0], torch.from_numpy(fixture["observation"]))
    assert torch.equal(robot.noise_scale(), torch.from_numpy(fixture["noise_scale"]))
    assert torch.equal(
        robot.previous_action_from_observation(observation[0]),
        torch.from_numpy(fixture["previous_action"]),
    )
    assert torch.equal(
        robot.pd_target(torch.from_numpy(fixture["action"]).unsqueeze(0), context)[0],
        torch.from_numpy(fixture["pd_target"]),
    )


def test_source_goal_and_actor_match_the_source_fixture_exactly():
    fixture = _fixture()
    robot, checkpoint = _robot()
    state = torch.from_numpy(fixture["state"]).unsqueeze(0)
    references = torch.from_numpy(fixture["references"]).unsqueeze(0)

    goal = source_goal_features(robot, references, state)

    assert goal.shape == (1, SOURCE_GOAL_DIM)
    assert torch.equal(goal[0], torch.from_numpy(fixture["goal"]))

    actor, world_model, critic = build_source_models(robot)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    world_model.load_state_dict(checkpoint["model"], strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    with torch.no_grad():
        action = actor(torch.from_numpy(fixture["history"]).unsqueeze(0), goal)

    assert action.shape == (1, SOURCE_ACTION_DIM)
    assert torch.equal(action[0], torch.from_numpy(fixture["action"]))


def test_source_normalization_vectors_match_the_registered_checkpoint():
    fixture = _fixture()
    robot, checkpoint = _robot()
    actor, world_model, critic = build_source_models(robot)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    world_model.load_state_dict(checkpoint["model"], strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)

    state_scale = torch.tensor([1.0] * 29 + [3.0] * 6 + [10.0] * 22 + [1.0] * 22)
    context_scale = torch.ones(SOURCE_CONTEXT_DIM)
    context_scale[120] = 0.016

    assert torch.equal(actor.obs_scale, checkpoint["actor"]["obs_scale"])
    assert torch.equal(actor.obs_scale[-6:], torch.ones(6))
    assert torch.equal(world_model.scale, state_scale)
    assert torch.equal(world_model.context_scale, context_scale)
    assert torch.equal(critic.scale, state_scale)
    assert world_model.representation == "heading"
    assert actor.encoder[0].in_features == SOURCE_HISTORY_FRAMES * 57
    assert actor.decoder[0].in_features == 32 + SOURCE_HISTORY_FRAMES * SOURCE_OBS_DIM


def test_source_profile_rejects_non_source_history_and_goal_frames():
    robot, _ = _robot()
    with pytest.raises(ValueError, match="10 frames by 78"):
        validate_source_history(torch.zeros(2, 9, SOURCE_OBS_DIM))
    with pytest.raises(ValueError, match="ten paired reference frames"):
        source_goal_features(
            robot, torch.zeros(1, 9, 79), torch.zeros(1, SOURCE_STATE_DIM)
        )
    robot.goal_representation = "absolute"
    with pytest.raises(ValueError, match="robot-anchored"):
        source_goal_features(
            robot, torch.zeros(1, 10, 79), torch.zeros(1, SOURCE_STATE_DIM)
        )


def test_source_profile_does_not_add_checkpoint_absent_modules():
    robot, _ = _robot()
    actor, _, _ = build_source_models(robot)

    assert not hasattr(actor, "speed_encoder")
    assert not hasattr(actor, "route_encoder")
    assert not hasattr(actor, "route_adapter")
    assert not any("speed" in name or "route" in name for name in actor.state_dict())
