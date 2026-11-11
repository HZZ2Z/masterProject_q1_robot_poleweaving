import math
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from expert.policy import ActorCritic
from pole_task import COURSE_ROOT, ROOT
from pole_task.data import CourseCase, PoleMotions, features_at, load_course, policy_course_features, write_vendor_motion
from pole_task.environment import PoleEnvironment
from pole_task.inference import load_pole_actor
from pole_task.models import (
    FEATURE_DIM,
    GOAL_DIM,
    PoleActor,
    PoleCritic,
    PoleRobot,
    PoleSuperMimic,
    PoleWorldModel,
    features_for_frames,
    observation_from_state,
    pole_goal_features,
    pole_actor_goal_features,
    route_condition,
)
from pole_task.preflight import preflight
from pole_task.replay import PoleReplay
from pole_task.scene import build_course_scene
from pole_task.training import (
    critic_and_model_gates,
    course_candidate_is_safe,
    guarded_actor_update,
    make_goal_provider,
    retention_candidate_is_safe,
    restore_resume_best_checkpoint,
)
from supermimic.algorithm import SuperMimic
from supermimic.models import Critic, WorldModel


def test_course_features_use_robot_frame_and_preserve_translation_invariance():
    reference = torch.zeros((1, 10, 29))
    reference[..., 3] = 1
    reference[..., 0] = 2
    poles = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    root = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([math.pi / 2])
    features = policy_course_features(reference, poles, root, yaw)

    assert features.shape == (1, FEATURE_DIM)
    assert torch.allclose(features[0, -6:-4], torch.tensor([0.0, -1.0]), atol=1e-6)

    shift = torch.tensor([[-2.0, 2.4]])
    shifted_reference = reference.clone()
    shifted_reference[..., :2] += shift[:, None, :]
    shifted = policy_course_features(shifted_reference, poles + shift[:, None], root + shift, yaw)
    assert torch.allclose(features, shifted, atol=1e-6)


def test_course_features_zero_retention_no_pole_sentinel():
    reference = torch.zeros((1, 10, 29))
    reference[..., 3] = 1
    poles = torch.tensor([[[100.0, 0.0], [102.0, 0.0], [104.0, 0.0]]])
    features = policy_course_features(
        reference, poles, torch.zeros((1, 2)), torch.zeros(1)
    )

    assert torch.equal(features[:, -6:], torch.zeros((1, 6)))


def test_course_case_applies_same_translation_to_reference_and_poles():
    original = CourseCase(
        "case", "train", np.array([[0, 0], [2, 0], [4, 0]], dtype=np.float32),
        np.tile(np.array([0, 0, 0.8, 1, 0, 0, 0] + [0] * 22, dtype=np.float32), (3, 1)),
        None, "fixture",
    )
    shifted = CourseCase(
        "case", "train", original.poles_xy + [-2.0, 2.4],
        original.qpos + np.array([-2.0, 2.4] + [0] * 27, dtype=np.float32),
        None, "fixture",
    )
    a = features_at(original, 0, torch.tensor([0.0, 0.0]), torch.tensor(0.0))
    b = features_at(shifted, 0, torch.tensor([-2.0, 2.4]), torch.tensor(0.0))
    assert torch.allclose(a, b, atol=1e-6)


def test_pole_actor_zero_initializes_route_path_and_preserves_expert_actions():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    expert = ActorCritic(cfg)
    actor = PoleActor(expert)
    observation = torch.randn(2, 79)
    history = torch.randn(2, 50, 79)
    features = torch.randn(2, GOAL_DIM)

    assert actor(observation, history, features).shape == (2, 22)
    assert torch.allclose(expert(observation, history), actor(observation, history, features), atol=1e-6, rtol=0)
    assert torch.count_nonzero(actor.route_residual[-1].weight) == 0


def test_pole_actor_training_scope_freezes_the_shared_expert_policy():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    actor = PoleActor(ActorCritic(cfg))

    assert not any(parameter.requires_grad for parameter in actor.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in actor.actor.parameters())
    assert all(parameter.requires_grad for parameter in actor.route_residual.parameters())
    assert [p for p in actor.parameters() if p.requires_grad] == list(actor.route_residual.parameters())


def test_pole_preflight_registers_source_warmup_without_loading_final_actor():
    from pole_task import ROOT

    report = preflight(COURSE_ROOT / "pole_task/config.json")
    assert report["initialization"]["source_checkpoint_id"] == "warmup_checkpoint"
    assert report["initialization"]["sha256"] == "b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5"
    assert report["initialization"]["available"]
    assert not report["initialization"]["source_final_checkpoint_loaded"]
    assert report["initialization"]["strictly_loaded"]
    assert report["training_ready"]
    assert report["training_blocker"] is None


def test_failing_pretraining_gate_disables_actor_update():
    config = {
        "held_out_model_rmse_max": 0.15,
        "held_out_termination_bce_max": 0.1,
        "critic_relative_rmse_max": 0.35,
    }
    metrics = {
        "held_out_transition_rmse": 0.151,
        "held_out_termination_bce": 0.05,
    }
    critic = {"critic_relative_rmse": 0.2}
    assert not critic_and_model_gates(metrics, critic, config)


def test_resume_carries_forward_the_previous_best_checkpoint(tmp_path):
    old_run = tmp_path / "old-run"
    new_run = tmp_path / "new-run"
    old_run.mkdir()
    new_run.mkdir()
    latest = old_run / "latest.pt"
    best = old_run / "best.pt"
    latest.write_bytes(b"latest")
    best.write_bytes(b"best-policy")

    copied = restore_resume_best_checkpoint(latest, new_run)

    assert copied == new_run / "best.pt"
    assert copied.read_bytes() == b"best-policy"


def test_failed_gate_never_calls_vendor_actor_update():
    class Algorithm:
        called = False

        def improve_policy(self, batch):
            self.called = True

    algorithm = Algorithm()
    result = guarded_actor_update(algorithm, object(), {"actor_updates_allowed": False})

    assert result is None
    assert not algorithm.called


def test_retention_regression_rejects_candidate_against_baseline():
    reports = {name: {"root_xy_rmse_m": 0.21, "fall": False} for name in ("straight", "left_turn", "right_turn")}
    baseline = {name: {"root_xy_rmse_m": 0.2} for name in reports}
    config = {"retention_case_ids": list(reports), "retention_rmse_m": 0.3}

    assert not retention_candidate_is_safe(reports, baseline, config)


def test_retention_warmup_allows_improving_but_not_yet_accepted_candidate():
    reports = {name: {"root_xy_rmse_m": 0.4, "fall": False} for name in ("straight", "left_turn", "right_turn")}
    baseline = {name: {"root_xy_rmse_m": 0.8} for name in reports}
    config = {"retention_case_ids": list(reports), "retention_rmse_m": 0.3}

    assert retention_candidate_is_safe(reports, baseline, config)


def test_course_regression_rejects_candidate_against_nominal_baseline():
    reports = {
        "nominal-01": {"root_xy_rmse_m": 0.4},
        "eval-01": {"root_xy_rmse_m": 0.4},
    }
    baseline = {
        "nominal-01": {"root_xy_rmse_m": 0.3},
        "eval-01": {"root_xy_rmse_m": 0.3},
    }

    assert not course_candidate_is_safe(reports, baseline, {"course_regression_tolerance_m": 0.05})


def test_course_gate_allows_small_warmup_regression_within_registered_tolerance():
    reports = {name: {"root_xy_rmse_m": 0.35} for name in ("nominal-01", "eval-01")}
    baseline = {name: {"root_xy_rmse_m": 0.3} for name in reports}

    assert course_candidate_is_safe(reports, baseline, {"course_regression_tolerance_m": 0.1})


def test_course_gate_compares_progress_when_baseline_terminates_early():
    reports = {
        "nominal-01": {"root_xy_rmse_m": 0.4, "steps": 675, "fall": False, "pole_collision": False, "route_completed": False},
        "eval-01": {"root_xy_rmse_m": 2.0, "steps": 671, "fall": False, "pole_collision": False, "route_completed": False},
    }
    baseline = {
        "nominal-01": {"root_xy_rmse_m": 0.3, "steps": 675, "fall": False, "pole_collision": False, "route_completed": False},
        "eval-01": {"root_xy_rmse_m": 1.7, "steps": 408, "fall": True, "pole_collision": False, "route_completed": False},
    }

    assert course_candidate_is_safe(reports, baseline, {"course_regression_tolerance_m": 0.1})


def test_vendor_reference_sequences_keep_each_course_paired_with_its_poles(tmp_path):
    nominal = load_course("nominal-01")
    heldout = load_course("eval-01")
    scene_path = build_course_scene(nominal, tmp_path / "course.xml")
    robot = PoleRobot(scene_path)
    paths = [
        write_vendor_motion(nominal, tmp_path / "nominal.npz"),
        write_vendor_motion(heldout, tmp_path / "heldout.npz"),
    ]
    motions = PoleMotions(robot, [nominal, heldout], paths, device="cpu")
    refs = motions.sequence(torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 1]))
    contexts = robot.context_for_poles(torch.tensor(np.stack([nominal.poles_xy, heldout.poles_xy])))

    assert torch.equal(refs[0, 0, 79:85], contexts[0, -6:])
    assert torch.equal(refs[1, 0, 79:85], contexts[1, -6:])
    assert not torch.equal(refs[0, 0, 79:85], refs[1, 0, 79:85])


def test_proprioception_keeps_the_22_action_history_slice():
    state = torch.randn(4, 79)
    observation = observation_from_state(state)
    assert observation.shape == (4, 79)
    assert torch.equal(observation[:, 57:79], state[:, 57:79])


def test_batched_reference_features_match_single_frame_features():
    case = CourseCase(
        "case", "train", np.array([[0, 0], [2, 0], [4, 0]], dtype=np.float32),
        np.tile(np.array([0, 0, 0.8, 1, 0, 0, 0] + [0] * 22, dtype=np.float32), (12, 1)),
        None, "fixture",
    )
    state = torch.zeros((2, 79))
    state[:, 2] = 0.8
    state[:, 3] = 1
    frames = torch.tensor([0, 3])

    features = features_for_frames(case, frames, state)

    expected = torch.stack([
        features_at(case, int(frame), state[index, :2], torch.tensor(0.0))
        for index, frame in enumerate(frames)
    ])
    assert features.shape == (2, FEATURE_DIM)
    assert torch.allclose(features, expected, atol=1e-6)


def test_pole_goal_features_use_live_robot_pose_instead_of_reference_pose(tmp_path):
    case = load_course("nominal-01")
    scene_path = build_course_scene(case, tmp_path / "course.xml")
    robot = PoleRobot(scene_path)
    references = torch.zeros((1, 10, 85))
    references[..., 3] = 1
    references[..., 79:85] = torch.tensor(case.poles_xy.reshape(1, 1, 6))
    state = torch.zeros((1, 109))
    state[:, 2] = 0.8
    state[:, 3] = 1
    state[:, :2] = torch.tensor(case.qpos[0, :2])

    expected = policy_course_features(
        references[:, :, :29], references[:, 0, 79:85].reshape(1, 3, 2),
        state[:, :2], torch.zeros(1),
    )[:, -6:]
    actual = pole_goal_features(robot, references, state)

    assert torch.allclose(actual[:, -6:], expected, atol=1e-6)
    translated_state = state.clone()
    translated_state[:, :2] += torch.tensor([1.0, 0.0])
    translated = pole_goal_features(robot, references, translated_state)
    assert not torch.allclose(actual[:, -6:], translated[:, -6:])


def test_route_condition_selects_next_pole_and_tracks_progress(tmp_path):
    case = load_course("nominal-01")
    robot = PoleRobot(build_course_scene(case, tmp_path / "course.xml"))
    references = torch.zeros((1, 10, 85))
    references[..., 3] = 1
    references[..., 79:85] = torch.tensor(case.poles_xy.reshape(1, 1, 6))
    state = torch.zeros((1, 109))
    state[:, 2] = 0.8
    state[:, 3] = 1
    state[:, 0] = float(case.poles_xy[0, 0] - 1.2)

    start = route_condition(robot, references, state)
    assert start.shape == (1, 7)
    assert torch.equal(start[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(start[0, 4], torch.tensor(0.0), atol=1e-6)

    state[:, 0] = float(case.poles_xy[0, 0] + 0.5)
    after_first = route_condition(robot, references, state)
    assert torch.equal(after_first[0, :3], torch.tensor([0.0, 1.0, 0.0]))
    assert after_first[0, 4] > start[0, 4]

    references[..., 79:85] = torch.tensor([100.0, 0.0, 102.0, 0.0, 104.0, 0.0])
    state[:, 1] = 0.5
    no_pole_features = route_condition(robot, references, state)
    assert torch.equal(no_pole_features[0, :5], torch.zeros(5))
    assert torch.allclose(no_pole_features[0, 5:], torch.tensor([-0.5, 0.0]))


def test_pole_reward_uses_whole_body_clearance_not_only_root(tmp_path):
    case = load_course("nominal-01")
    robot = PoleRobot(build_course_scene(case, tmp_path / "course.xml"))
    no_pole_context = robot.context_for_poles(
        torch.tensor([[[100.0, 0.0], [102.0, 0.0], [104.0, 0.0]]])
    )
    assert torch.equal(no_pole_context[:, -6:], torch.zeros((1, 6)))
    qpos = torch.tensor(case.qpos[0]).unsqueeze(0)
    state = torch.cat((qpos, torch.zeros((1, 50))), dim=-1)
    bodies = robot.kinematics(state)[0]
    target = torch.zeros((1, 85))
    target[:, :29] = qpos
    target[:, 3] = 1
    target[:, 79:85] = torch.tensor([[100.0, 0.0, 102.0, 0.0, 104.0, 0.0]])
    action = torch.zeros((1, 22))
    clear_reward = robot.reward(state, state, target, action, 0.02)
    target[:, 79:81] = bodies[:, 8, :2]
    near_link_reward = robot.reward(state, state, target, action, 0.02)

    assert near_link_reward < clear_reward


def test_pole_models_and_algorithm_extend_the_vendored_supermimic_core():
    assert issubclass(PoleWorldModel, WorldModel)
    assert issubclass(PoleCritic, Critic)
    assert issubclass(PoleRobot, __import__("supermimic.robot", fromlist=["Q1Robot"]).Q1Robot)
    assert issubclass(PoleSuperMimic, SuperMimic)


def test_pole_actor_implements_supermimic_goal_encode_decode_contract():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    expert = ActorCritic(cfg)
    actor = PoleActor(expert)
    history = torch.randn(3, 50, 79)
    goal = torch.randn(3, GOAL_DIM)

    encoding = actor.encode_goal(goal)
    action = actor.decode(history, encoding)

    assert encoding.shape == (3, actor.goal_embedding_dim)
    assert action.shape == (3, 22)
    assert torch.allclose(
        actor.decode(history, torch.zeros_like(encoding)),
        expert(history[:, 0], history),
        atol=1e-6,
        rtol=0,
    )


def test_control_and_supermimic_paths_share_the_trained_goal_encoder():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    actor = PoleActor(ActorCritic(cfg))
    with torch.no_grad():
        for parameter in actor.route_residual.parameters():
            parameter.normal_(0, 0.1)
    history = torch.randn(2, 50, 79)
    observation = history[:, 0]
    goal = torch.randn(2, GOAL_DIM)

    direct = actor.forward_control(observation, history, goal)
    trained = actor(history, goal)

    assert torch.allclose(direct, trained, atol=1e-6, rtol=0)


def test_route_residual_uses_tracking_but_ignores_retention_sentinel_poles():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    expert = ActorCritic(cfg)
    actor = PoleActor(expert)
    with torch.no_grad():
        for parameter in actor.route_residual.parameters():
            parameter.normal_(0, 0.1)
    observation = torch.randn(2, 79)
    history = torch.randn(2, 50, 79)
    goal = torch.randn(2, GOAL_DIM)
    goal[:, -13:-7] = 100

    actual = actor.forward_control(observation, history, goal)
    expected = expert(observation, history)

    encoded = actor.encode_goal(goal)
    assert torch.equal(encoded[:, :6], torch.zeros((2, 6)))
    assert torch.equal(encoded[:, 6:], goal[:, -7:])
    assert not torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_pole_actor_clips_actions_to_environment_bounds():
    cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    expert = ActorCritic(cfg)
    with torch.no_grad():
        expert.actor[-1].bias.fill_(2.0)
    actor = PoleActor(expert)
    observation = torch.zeros(1, 79)
    history = torch.zeros(1, 50, 79)
    goal = torch.zeros(1, GOAL_DIM)
    goal[:, -13:-7] = 1.0

    action = actor.forward_control(observation, history, goal)
    decoded = actor.decode(history, actor.encode_goal(goal))

    assert action.max() <= 1.0 and action.min() >= -1.0
    assert torch.equal(action, decoded)


def test_training_case_selection_includes_retention_but_excludes_held_out():
    nominal = SimpleNamespace(case_id="nominal-01", split="train")
    heldout = SimpleNamespace(case_id="eval-01", split="held_out")
    retention = [
        SimpleNamespace(case_id=name, split="retention")
        for name in ("straight", "left_turn", "right_turn")
    ]
    config = json.loads((COURSE_ROOT / "pole_task/config.json").read_text())
    assert config["training_case_ids"] == [
        "nominal-01", "straight", "left_turn", "right_turn"
    ]
    from pole_task.training import select_training_cases

    selected = select_training_cases([nominal, heldout, *retention], config)

    assert [case.case_id for case in selected] == [
        "nominal-01", "straight", "left_turn", "right_turn"
    ]
    with pytest.raises(ValueError, match="[Hh]eld-out"):
        select_training_cases([nominal, heldout], {"training_case_ids": ["eval-01"]})


def test_pole_motions_preserve_course_identity_in_vendor_reference_batches(tmp_path):
    case = load_course("nominal-01")
    scene_path = build_course_scene(case, tmp_path / "course.xml")
    robot = PoleRobot(scene_path)
    motion_path = write_vendor_motion(case, tmp_path / "nominal.npz")
    motions = PoleMotions(robot, [case], [motion_path], device="cpu")
    motions.prepare()

    sequence = motions.sequence(torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 1]))

    assert sequence.shape == (1, 2, 85)
    assert torch.allclose(sequence[0, :, 79:85], torch.tensor(case.poles_xy.reshape(1, 6)).expand(2, -1))
    assert torch.allclose(sequence[0, 0, :2], torch.tensor(case.qpos[0, :2]))
    assert motions.remaining(torch.tensor([0]), torch.tensor([0])).item() == 675

    provider = make_goal_provider(robot, motions, [case], "cpu")
    state = torch.zeros((1, 109))
    state[:, 2] = 0.8
    state[:, 3] = 1
    state[:, :2] = torch.tensor(case.qpos[0, :2])
    goal = provider(case, 0, state)
    assert goal.shape == (1, GOAL_DIM)


def test_environment_state_uses_vendor_world_angular_velocity_and_matches_expert_obs(tmp_path):
    case = load_course("nominal-01")
    env = PoleEnvironment(case, tmp_path / "course.xml")
    env.reset()
    env.data.qpos[3:7] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    env.data.qvel[3:6] = [1.0, 0.0, 0.0]
    state = env.state()

    assert np.allclose(state[32:35], [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(observation_from_state(torch.from_numpy(state), torch.tensor(env.default)).numpy(), env.observation(), atol=1e-6)


def test_near_pole_reward_is_finite_and_real_contact_terminates_episode(tmp_path):
    base = load_course("nominal-01")
    near_poles = base.poles_xy.copy()
    near_poles[0] = base.qpos[0, :2] + [0.24, 0.0]
    near_case = CourseCase("near", "train", near_poles, base.qpos, base.path, base.sha256)
    near_env = PoleEnvironment(near_case, tmp_path / "near.xml")
    near_env.reset()
    _, near_reward, _, _ = near_env.step(np.zeros(22, dtype=np.float32))
    assert np.isfinite(near_reward)

    contact_poles = np.repeat(base.qpos[0, None, :2], 3, axis=0)
    contact_case = CourseCase("contact", "train", contact_poles, base.qpos, base.path, base.sha256)
    contact_env = PoleEnvironment(contact_case, tmp_path / "contact.xml")
    contact_env.reset()
    _, _, done, transition = contact_env.step(np.zeros(22, dtype=np.float32))
    assert done
    assert transition["terminal_cause"] == "pole_collision"
    assert transition["pole_collision"]
    assert not contact_env.result()["success"]


def test_course_result_fields_cover_success_collision_and_fall(tmp_path):
    case = load_course("nominal-01")
    env = PoleEnvironment(case, tmp_path / "course.xml")
    env.reset()
    env.data.qpos[:2] = case.qpos[-1, :2]
    env.frame = len(case.qpos) - 1
    env.track_sq_error = [0.0]
    success = env.result()
    assert success["steps"] == 0
    assert success["endpoint_distance_m"] <= 0.5
    assert success["route_completed"] and success["success"]

    env.collision = True
    collision = env.result()
    assert collision["pole_collision"] and not collision["success"]
    env.collision = False
    env.fall = True
    fall = env.result()
    assert fall["fall"] and not fall["success"]


def test_playback_loader_rejects_missing_soccer_and_source_final_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pole checkpoint not found"):
        load_pole_actor(tmp_path / "missing.pt")
    soccer_checkpoint = tmp_path / "soccer.pt"
    torch.save({"format": "primebot_supermimic_soccer_v1"}, soccer_checkpoint)
    with pytest.raises(ValueError, match="Soccer checkpoint cannot drive pole playback"):
        load_pole_actor(soccer_checkpoint)
    source_final = tmp_path / "source-final.pt"
    torch.save(
        {"config": {}, "identity": {"checkpoint_id": "passed_final_checkpoint_reference"}, "iteration": 2500},
        source_final,
    )
    with pytest.raises(ValueError, match="Source task 7.7 checkpoint is a reference artifact"):
        load_pole_actor(source_final)


def test_playback_loader_rejects_a_soccer_profile_identity(tmp_path):
    wrong_profile = tmp_path / "wrong-profile.pt"
    torch.save(
        {
            "format": "supermimic_pole_weaving_profile_v1",
            "identity": {
                "profile": "soccer",
                "source_final_checkpoint_loaded": False,
            },
        },
        wrong_profile,
    )
    with pytest.raises(ValueError, match="profile 'soccer' is not pole-weaving"):
        load_pole_actor(wrong_profile)


def test_playback_loader_rejects_stale_reference_frame_pole_checkpoint(tmp_path):
    stale_checkpoint = tmp_path / "stale-v3.pt"
    torch.save({
        "format": "q1_pole_weaving_project_v3",
        "identity": {"source_task_actor_loaded": False},
    }, stale_checkpoint)

    with pytest.raises(ValueError, match="not a project-trained pole policy"):
        load_pole_actor(stale_checkpoint)


def test_playback_loader_rejects_old_retention_sentinel_checkpoint(tmp_path):
    stale_checkpoint = tmp_path / "retention-sentinel-v8.pt"
    torch.save({
        "format": "q1_pole_weaving_project_v8",
        "identity": {"source_task_actor_loaded": False},
    }, stale_checkpoint)

    with pytest.raises(ValueError, match="not a project-trained pole policy"):
        load_pole_actor(stale_checkpoint)


def test_vendor_replay_returns_project_history_in_newest_first_order():
    replay = PoleReplay(capacity=16, num_envs=1, state_dim=79, context_dim=4, obs_dim=79, device="cpu")
    for frame in range(8):
        clip = frame // 4
        local_frame = frame % 4
        state = torch.zeros(1, 79)
        observation = torch.full((1, 79), float(frame))
        replay.add(
            states=state, next_states=state, actions=torch.zeros(1, 22),
            context=torch.tensor([[clip, 2.0, 3.0, 4.0]]), observation=observation,
            terminated=torch.zeros(1), boundary=torch.tensor([local_frame == 3]),
            clips=torch.tensor([clip]), frames=torch.tensor([local_frame]),
        )
    batch = replay.sample(batch_size=1, window=1, history=3)

    assert batch.history[0, 0, 0] > batch.history[0, -1, 0]
    assert torch.equal(batch.context[:, 0].long(), batch.clips)
    assert batch.frames[0] in (2, 3)


def test_vendor_imagination_receives_reference_and_pole_conditioned_goal(tmp_path):
    case = load_course("nominal-01")
    scene_path = build_course_scene(case, tmp_path / "course.xml")
    robot = PoleRobot(scene_path)
    motion_path = write_vendor_motion(case, tmp_path / "nominal.npz")
    motions = PoleMotions(robot, [case], [motion_path], device="cpu")
    actor_cfg = {
        "env": {"num_critic_obs": 91, "num_observations": 79, "num_actions": 22, "num_privileged_obs": 14},
        "algorithm": {"history_embedding_dim": 64},
        "runner": {"num_stack": 50},
    }
    actor = PoleActor(ActorCritic(actor_cfg))
    algorithm = PoleSuperMimic(
        robot, motions, PoleWorldModel(robot, (32, 32, 32)), actor, PoleCritic(robot, (32, 16)),
        learning_rate=1e-5, horizon=2,
    )
    clips, frames = torch.tensor([0]), torch.tensor([0])
    references = motions.sequence(clips, frames, torch.arange(10))
    state = references[:, 0, :79]
    poles = torch.tensor(case.poles_xy).unsqueeze(0)
    context = robot.context_for_poles(poles)
    history = torch.zeros(1, 50, 79)

    next_state, reward, terminal, observation, goal, value = algorithm._imagined_step(
        state, history, references, context
    )

    assert next_state.shape == state.shape
    assert observation.shape == (1, 79)
    assert goal.shape == (1, GOAL_DIM)
    assert value.shape == (1,)
    assert torch.isfinite(reward).all() and torch.isfinite(terminal).all()

    replay = PoleReplay(96, 1, 79, robot.context_dim, 79, "cpu")
    context = robot.context_for_poles(poles)
    for frame in range(80):
        pair = motions.sequence(clips, torch.tensor([frame]), torch.tensor([0, 1]))[0]
        state_row, next_row = pair[0, :79].unsqueeze(0), pair[1, :79].unsqueeze(0)
        observation_row = robot.observation(state_row, context)
        replay.add(
            state_row, next_row, torch.zeros(1, 22), context, observation_row,
            torch.zeros(1), torch.zeros(1, dtype=torch.bool), clips,
            torch.tensor([frame]),
        )
    model_metrics = algorithm.fit_model(replay.sample(2, window=8, history=1))
    assert torch.isfinite(torch.tensor(model_metrics["model_loss"]))
    assert torch.isfinite(torch.tensor(model_metrics["termination_loss"]))
    frozen_before = [parameter.detach().clone() for parameter in (*actor.encoder.parameters(), *actor.actor.parameters())]
    residual_before = [parameter.detach().clone() for parameter in actor.route_residual.parameters()]
    policy_metrics = algorithm.improve_policy(
        replay.sample(1, window=1, history=50), critic_epochs=1, critic_minibatches=1
    )
    assert torch.isfinite(torch.tensor(policy_metrics["actor_loss"]))
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(
            frozen_before, (*actor.encoder.parameters(), *actor.actor.parameters()), strict=True
        )
    )
    assert any(
        not torch.equal(before, parameter)
        for before, parameter in zip(residual_before, actor.route_residual.parameters(), strict=True)
    )
