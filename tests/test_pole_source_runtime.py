"""Source-profile rollout, replay, imagination, and evaluation identity."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch
from torch import nn

from pole_task import COURSE_ROOT, ROOT
from pole_task.data import CourseCase, PoleMotions, load_course, write_vendor_motion
from pole_task.models import build_source_models
from pole_task.scene import build_course_scene
from pole_task.source import source_robot_xml
from pole_task.source_environment import SourcePoleEnvironment
from pole_task.source_profile import (
    SOURCE_GOAL_DIM,
    SOURCE_OBS_DIM,
    SourcePoleRobot,
    source_goal_features,
)
from pole_task.source_runtime import (
    SOURCE_REWARD_TUNING,
    SourcePoleSuperMimic,
    _episode_result,
    evaluate_source_cases,
    make_source_replay,
    run_source_episode,
    source_batch_goals,
    source_batch_references,
    source_nominal_context,
)
from pole_task.training import case_from_retention

SOURCE_MANIFEST = COURSE_ROOT / "pole_task/source_manifest.json"


class ZeroActor(nn.Module):
    def forward(self, history, goal):
        return history.new_zeros((history.shape[0], 22))


def _robot() -> SourcePoleRobot:
    robot = SourcePoleRobot(source_robot_xml(SOURCE_MANIFEST))
    robot.set_course_frame((0.0, 0.0, 0.0), 0.0, 0.04)
    return robot


def _motions(tmp_path, cases):
    robot = _robot()
    paths = [
        write_vendor_motion(case, tmp_path / f"{case.case_id}.npz") for case in cases
    ]
    motions = PoleMotions(robot, cases, paths, device="cpu")
    motions.prepare()
    return robot, motions


def test_reset_and_rollout_keep_paired_course_and_reference_identity(tmp_path):
    nominal, retention = load_course("nominal-01"), case_from_retention("straight")
    robot, motions = _motions(tmp_path, [nominal, retention])

    episode = run_source_episode(
        ZeroActor(),
        robot,
        motions,
        nominal,
        clip=0,
        scene_path=tmp_path / "nominal.xml",
        start_frame=5,
        seed=11,
        max_steps=3,
    )

    assert episode.case.identity == nominal.identity
    assert np.array_equal(episode.states[0, :29], nominal.qpos[5])
    expected_context = source_nominal_context(
        robot,
        torch.as_tensor(nominal.poles_xy, dtype=torch.float32).unsqueeze(0),
    )
    expected_observation = robot.observation(
        torch.as_tensor(episode.states[0], dtype=torch.float32).unsqueeze(0),
        expected_context,
    )[0].numpy()
    assert np.allclose(episode.observations[0], expected_observation, atol=1e-6)

    assert episode.frames.tolist() == [5, 6, 7]
    assert episode.clip == 0
    assert np.allclose(episode.actions, 0.0)
    for transition in episode.transitions:
        assert transition["reference_id"] == nominal.identity
        assert np.array_equal(transition["pole_xy"], nominal.poles_xy)
        stored = np.asarray(transition["context"])[
            robot.context_layout.course_poles
        ]
        assert np.allclose(stored, nominal.poles_xy.reshape(-1))
        references = motions.sequence(
            torch.tensor([0]),
            torch.tensor([transition["frame"]]),
            torch.arange(1, 11),
        )
        target = nominal.qpos[transition["frame"] + 1]
        assert np.allclose(references[0, 0, :2], target[:2])
        assert np.allclose(references[0, 0, 7:29], target[7:29])
    # Truncation closes the episode so replay windows cannot splice courses.
    assert episode.boundaries[-1]
    assert episode.transitions[-1]["boundary"]


def test_replay_sampling_preserves_reference_and_course_pairing(tmp_path):
    nominal = load_course("nominal-01")
    retention = case_from_retention("straight")
    robot, motions = _motions(tmp_path, [nominal, retention])
    episodes = [
        run_source_episode(
            ZeroActor(),
            robot,
            motions,
            case,
            clip=index,
            scene_path=tmp_path / f"{case.case_id}.xml",
            start_frame=0,
            seed=100 + index,
            max_steps=16,
        )
        for index, case in enumerate((nominal, retention))
    ]
    clip_by_case = {case.case_id: index for index, case in enumerate((nominal, retention))}
    replay = make_source_replay(episodes, robot, clip_by_case, device="cpu")

    observations = {
        (episode.clip, int(transition["frame"])): transition["observation"]
        for episode in episodes
        for transition in episode.transitions
    }
    batch = replay.sample(batch_size=16, window=1, history=10)

    assert batch.history.shape == (16, 10, SOURCE_OBS_DIM)
    assert batch.context.shape[-1] == robot.context_dim
    expected_poles = motions.poles_xy[batch.clips]
    stored_poles = batch.context[:, robot.context_layout.course_poles].reshape(-1, 3, 2)
    assert torch.allclose(stored_poles, expected_poles)
    assert source_batch_goals(robot, motions, batch).shape == (16, SOURCE_GOAL_DIM)

    references = source_batch_references(
        motions, batch, torch.arange(1, 11)
    )
    for row in range(len(batch.clips)):
        case = motions.cases[int(batch.clips[row])]
        frame = int(batch.frames[row])
        assert case.case_id != "eval-01"
        assert np.allclose(
            references[row, 0, :2].numpy(), case.qpos[frame + 1, :2]
        )
        assert np.allclose(
            references[row, 0, 7:29].numpy(), case.qpos[frame + 1, 7:29]
        )
        newest = observations[(int(batch.clips[row]), frame)]
        assert np.allclose(batch.history[row, -1].numpy(), newest)
        # Oldest-to-newest source order: ten consecutive frames ending here.
        episode = episodes[int(batch.clips[row])]
        assert frame >= 9
        assert np.allclose(batch.history[row, 0].numpy(), episode.observations[frame - 9])


def test_imagination_uses_live_state_goals_and_paired_course_context(tmp_path):
    case = load_course("nominal-01")
    robot, motions = _motions(tmp_path, [case])
    actor, world, critic = build_source_models(robot)
    algorithm = SourcePoleSuperMimic(
        robot,
        motions,
        world,
        actor,
        critic,
        learning_rate=1e-5,
        horizon=2,
        reward_tuning=SOURCE_REWARD_TUNING,
        clean_policy_actions=True,
        normalize_policy_objective=True,
    )

    from supermimic.replay import TemporalReplay

    replay = TemporalReplay(
        capacity=64,
        num_envs=1,
        state_dim=79,
        context_dim=robot.context_dim,
        obs_dim=SOURCE_OBS_DIM,
        device="cpu",
    )
    poles = torch.as_tensor(case.poles_xy, dtype=torch.float32).unsqueeze(0)
    context = source_nominal_context(robot, poles)
    for frame in range(40):
        states = motions.sequence(
            torch.tensor([0]), torch.tensor([frame]), torch.tensor([0, 1])
        )[0]
        state, next_state = states[0, :79], states[1, :79]
        observation = robot.observation(state.unsqueeze(0), context)
        replay.add(
            state.unsqueeze(0),
            next_state.unsqueeze(0),
            torch.zeros(1, 22),
            context,
            observation,
            torch.zeros(1),
            torch.tensor([frame == 39]),
            torch.tensor([0]),
            torch.tensor([frame]),
        )
    batch = replay.sample(batch_size=4, window=1, history=10)

    references = source_batch_references(
        motions, batch, torch.arange(1, 11)
    )
    expected_goal = source_goal_features(robot, references, batch.states[:, 0])
    next_state, reward, termination, observation, goal, value = algorithm._imagined_step(
        batch.states[:, 0], batch.history, references, batch.context
    )

    assert torch.equal(goal, expected_goal)
    assert next_state.shape == batch.states[:, 0].shape
    assert reward.shape == (4,) and termination.shape == (4,)
    assert observation.shape == (4, SOURCE_OBS_DIM)
    assert value.shape == (4,)
    assert torch.isfinite(reward).all() and torch.isfinite(termination).all()
    stored_poles = algorithm.model.advance_context(batch.context, next_state[:, 57:79])
    assert torch.allclose(
        stored_poles[:, robot.context_layout.course_poles],
        batch.context[:, robot.context_layout.course_poles],
    )


def test_evaluation_reports_registered_metrics_and_real_pole_contact(tmp_path):
    base = load_course("nominal-01")
    short = CourseCase(
        "short-nominal",
        "train",
        base.poles_xy,
        base.qpos[:20],
        base.path,
        base.sha256,
    )
    robot, motions = _motions(tmp_path, [short])
    reports = evaluate_source_cases(
        ZeroActor(),
        robot,
        motions,
        [short],
        tmp_path / "evaluation.xml",
        seed=7,
    )
    row = reports["short-nominal"]
    assert row["case_id"] == "short-nominal"
    assert row["reference_id"] == short.identity
    assert row["steps"] == 19
    assert {
        "endpoint_distance_m",
        "route_completed",
        "pole_collision",
        "pole_contacts",
        "fall",
        "root_xy_rmse_m",
        "success",
    } <= set(row)

    contact_poles = np.repeat(base.qpos[0, None, :2], 3, axis=0)
    contact = CourseCase(
        "contact", "train", contact_poles, base.qpos[:20], base.path, base.sha256
    )
    robot, contact_motions = _motions(tmp_path / "contact", [contact])
    episode = run_source_episode(
        ZeroActor(),
        robot,
        contact_motions,
        contact,
        clip=0,
        scene_path=tmp_path / "contact-scene.xml",
        start_frame=0,
        seed=3,
        max_steps=5,
    )
    assert episode.result["pole_collision"]
    assert episode.result["pole_contacts"]
    assert not episode.result["success"]
    assert not episode.result["route_completed"]
    assert episode.transitions[0]["terminal_cause"] == "pole_collision"


def test_episode_metrics_are_consistent_for_success_collision_fall_and_early_stop():
    base = load_course("nominal-01")

    class FakeEnvironment:
        def __init__(self, frame, collision=False):
            self.frame = frame
            self.collision = collision

    frame_count = 2
    case = CourseCase(
        "metric-case", "train", base.poles_xy, base.qpos[:frame_count], base.path, base.sha256
    )
    states = [case.qpos[0], case.qpos[1]]
    transitions = [{"frame": 0, "pole_contacts": []}]
    expected_fields = {
        "case_id",
        "reference_id",
        "steps",
        "endpoint_distance_m",
        "route_completed",
        "pole_collision",
        "pole_contacts",
        "fall",
        "root_xy_rmse_m",
        "success",
    }

    success = _episode_result(
        case, FakeEnvironment(1), states, transitions, steps=1, fall=False
    )
    assert set(success) == expected_fields
    assert success["steps"] == 1
    assert success["route_completed"] and success["success"]

    collision = _episode_result(
        case, FakeEnvironment(1, collision=True), states, transitions, steps=1, fall=False
    )
    assert set(collision) == expected_fields
    assert collision["pole_collision"] and not collision["success"]

    fall = _episode_result(
        case, FakeEnvironment(1), states, transitions, steps=1, fall=True
    )
    assert set(fall) == expected_fields
    assert fall["fall"] and not fall["success"]

    early_case = CourseCase(
        "early-stop", "train", base.poles_xy, base.qpos[:5], base.path, base.sha256
    )
    early_states = [early_case.qpos[0], early_case.qpos[1], early_case.qpos[2]]
    early = _episode_result(
        early_case,
        FakeEnvironment(2),
        early_states,
        [{"frame": 0, "pole_contacts": []}, {"frame": 1, "pole_contacts": []}],
        steps=2,
        fall=False,
    )
    assert set(early) == expected_fields
    assert early["steps"] == 2
    assert not early["route_completed"] and not early["success"]

    contacts = _episode_result(
        case,
        FakeEnvironment(1, collision=True),
        states,
        [{"frame": 0, "pole_contacts": ["pole_2", "pole_1", "pole_2"]}],
        steps=1,
        fall=False,
    )
    assert contacts["pole_contacts"] == ["pole_1", "pole_2"]


def test_randomized_reset_applies_source_context_and_is_seed_deterministic(tmp_path):
    nominal = load_course("nominal-01")
    robot, _ = _motions(tmp_path, [nominal])
    layout = robot.context_layout
    environment = SourcePoleEnvironment(
        nominal, tmp_path / "randomized.xml", robot, seed=21, randomize=True
    )
    base_mass = environment.default_mass.copy()
    base_ipos = environment.default_ipos.copy()
    base_friction = environment.default_friction.copy()

    environment.reset(5)
    context = environment.context[0]
    assert 0.3 <= float(context[layout.friction]) <= 1.2
    assert np.all(context[layout.kp_factor].numpy() >= 0.8)
    assert np.all(context[layout.kp_factor].numpy() <= 1.2)
    assert np.all(context[layout.kd_factor].numpy() >= 0.9)
    assert np.all(context[layout.kd_factor].numpy() <= 1.1)
    assert np.all((context[layout.motor_offset].numpy() >= -0.02))
    assert np.all((context[layout.motor_offset].numpy() <= 0.02))
    assert -1.5 <= float(context[layout.torso_added_mass]) <= 1.5
    assert np.array_equal(
        context[layout.course_poles].numpy().reshape(3, 2), nominal.poles_xy
    )
    assert float(context[layout.delay_s]) == 0.0
    torso = environment.torso_body
    assert environment.model.body_mass[torso] == pytest.approx(
        base_mass[torso] + float(context[layout.torso_added_mass])
    )
    assert np.allclose(
        environment.model.body_ipos[environment.root_body],
        base_ipos[environment.root_body] + context[layout.root_com].numpy(),
    )
    assert np.allclose(
        environment.model.geom_friction[environment.foot_geoms, 0],
        float(context[layout.friction]),
    )
    assert (
        environment.model.geom_friction[environment.floor_geoms, 0] <= 0.3
    ).all()
    # Every non-contact friction coefficient remains the XML default.
    assert np.array_equal(
        environment.model.geom_friction[environment.floor_geoms, 1:],
        base_friction[environment.floor_geoms, 1:],
    )
    assert np.allclose(
        environment.zero,
        robot.default_q.numpy() + context[layout.motor_offset].numpy(),
    )
    assert np.allclose(
        environment.kp_effective,
        environment.kp * context[layout.kp_factor].numpy(),
    )
    assert np.allclose(
        environment.kd_effective,
        environment.kd * context[layout.kd_factor].numpy(),
    )
    first_context = context.clone()
    first_state = environment.state()
    environment.reset(5)
    # A reset reseeds startup randomization deterministically.
    assert not torch.equal(first_context, environment.context[0])

    replay = SourcePoleEnvironment(
        nominal, tmp_path / "replay.xml", robot, seed=21, randomize=True
    )
    replay.reset(5)
    assert torch.equal(first_context, replay.context[0])
    assert np.array_equal(first_state, replay.state())

    nominal_environment = SourcePoleEnvironment(
        nominal, tmp_path / "nominal.xml", robot, seed=21
    )
    nominal_environment.reset(5)
    nominal_context = nominal_environment.context[0]
    assert float(nominal_context[layout.friction]) == pytest.approx(1.0)
    assert torch.allclose(
        nominal_context[layout.kp_factor], torch.ones_like(nominal_context[layout.kp_factor])
    )
    assert torch.allclose(
        nominal_context[layout.kd_factor], torch.ones_like(nominal_context[layout.kd_factor])
    )
    assert np.array_equal(
        nominal_environment.model.body_mass, environment.default_mass
    )


def test_randomized_push_marks_a_replay_boundary(tmp_path):
    nominal = load_course("nominal-01")
    robot, motions = _motions(tmp_path, [nominal])
    environment = SourcePoleEnvironment(
        nominal, tmp_path / "push.xml", robot, seed=4, randomize=True
    )
    environment.reset(5)
    environment.push_timer = 0.0
    velocity_before = environment.data.qvel.copy()
    _, transition = environment.step(np.zeros(22))
    assert transition["pushed"]
    assert environment.pushed
    assert not np.array_equal(velocity_before, environment.data.qvel)

    episode = run_source_episode(
        ZeroActor(),
        robot,
        motions,
        nominal,
        clip=0,
        scene_path=tmp_path / "randomized-rollout.xml",
        start_frame=5,
        seed=4,
        max_steps=3,
        randomize=True,
    )
    layout = robot.context_layout
    stored = torch.as_tensor(episode.contexts[0])
    assert np.array_equal(
        stored[layout.course_poles].numpy().reshape(3, 2), nominal.poles_xy
    )
    assert 0.3 <= float(stored[layout.friction]) <= 1.2


SOCCER_PRESENTATION_GEOMS = (
    "soccer_ball",
    "goal_left_post",
    "goal_right_post",
    "goal_back",
    "opponent_goal_left_post",
    "opponent_goal_right_post",
    "opponent_goal_back",
)


def test_source_rollout_ignores_ball_and_goal_behavior(tmp_path):
    nominal = load_course("nominal-01")
    robot, motions = _motions(tmp_path, [nominal])
    baseline = build_course_scene(nominal, tmp_path / "baseline.xml")
    shifted = build_course_scene(nominal, tmp_path / "shifted.xml")

    start = nominal.qpos[5]
    tree = ET.parse(shifted)
    root = tree.getroot()
    for name in SOCCER_PRESENTATION_GEOMS:
        geom = root.find(f".//geom[@name='{name}']")
        assert geom is not None, f"missing soccer presentation geom {name}"
        geom.set("pos", f"{start[0]:.9g} {start[1]:.9g} 0.5")
    tree.write(shifted, encoding="utf-8", xml_declaration=True)

    def rollout(scene_path):
        return run_source_episode(
            ZeroActor(),
            robot,
            motions,
            nominal,
            clip=0,
            scene_path=scene_path,
            start_frame=5,
            seed=11,
            max_steps=12,
        )

    base = rollout(baseline)
    moved = rollout(shifted)
    # Ball and goals are outside observations, rewards, reset, and termination:
    # putting every soccer geom on the robot must not change one rollout value.
    assert np.array_equal(base.states, moved.states)
    assert np.array_equal(base.observations, moved.observations)
    assert np.array_equal(base.actions, moved.actions)
    assert np.array_equal(base.rewards, moved.rewards)
    assert np.array_equal(base.terminated, moved.terminated)
    assert np.array_equal(base.boundaries, moved.boundaries)
    assert [row["terminal_cause"] for row in base.transitions] == [
        row["terminal_cause"] for row in moved.transitions
    ]
    assert [row["pole_contacts"] for row in base.transitions] == [
        row["pole_contacts"] for row in moved.transitions
    ]

    environment = SourcePoleEnvironment(
        nominal, tmp_path / "inspection.xml", robot, seed=11
    )
    model = environment.model
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") == -1
    assert model.nq == len(nominal.qpos[0])
    ball = model.geom("soccer_ball").id
    assert model.geom_bodyid[ball] == 0
    presentation = {
        model.geom(name).id for name in SOCCER_PRESENTATION_GEOMS
    }
    environment.reset(5)
    ball_pose = environment.data.geom_xpos[ball].copy()
    for _ in range(12):
        environment.step(np.zeros(22))
        # The ball has no body, joint, or degree of freedom: it stays scenery.
        assert np.array_equal(environment.data.geom_xpos[ball], ball_pose)
        for contact in environment.data.contact[: environment.data.ncon]:
            assert contact.geom1 not in presentation
            assert contact.geom2 not in presentation


def test_source_rollout_does_not_invoke_soccer_policy(tmp_path, monkeypatch):
    from expert.policy import ActorCritic

    nominal = load_course("nominal-01")
    robot, motions = _motions(tmp_path, [nominal])

    def blocked(*args, **kwargs):
        raise AssertionError("soccer policy invoked during a pole rollout")

    monkeypatch.setattr(ActorCritic, "forward", blocked)
    episode = run_source_episode(
        ZeroActor(),
        robot,
        motions,
        nominal,
        clip=0,
        scene_path=tmp_path / "policy.xml",
        start_frame=0,
        max_steps=3,
    )
    assert episode.steps == 3


def _parameters(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def test_model_fit_and_actor_update_use_paired_course_context(tmp_path):
    nominal = load_course("nominal-01")
    robot, motions = _motions(tmp_path, [nominal])
    episode = run_source_episode(
        ZeroActor(),
        robot,
        motions,
        nominal,
        clip=0,
        scene_path=tmp_path / "training.xml",
        start_frame=0,
        seed=5,
        max_steps=24,
    )
    assert episode.steps == 24
    replay = make_source_replay(
        [episode], robot, {nominal.case_id: 0}, device="cpu"
    )
    actor, world, critic = build_source_models(robot)
    algorithm = SourcePoleSuperMimic(
        robot,
        motions,
        world,
        actor,
        critic,
        learning_rate=1e-4,
        horizon=2,
        reward_tuning=SOURCE_REWARD_TUNING,
        clean_policy_actions=True,
        normalize_policy_objective=True,
    )
    expected = torch.as_tensor(
        nominal.poles_xy.reshape(-1), dtype=torch.float32
    )
    columns = robot.context_layout.course_poles

    model_batch = replay.sample(batch_size=8, window=8, history=1)
    assert torch.allclose(model_batch.context[:, columns], expected)
    context_before = model_batch.context.clone()
    world_before = _parameters(world)
    model_metrics = algorithm.fit_model(model_batch)
    assert set(model_metrics) == {"model_loss", "state_loss", "termination_loss"}
    assert all(np.isfinite(value) for value in model_metrics.values())
    assert not torch.equal(world_before, _parameters(world))
    # Model fitting advances the delay buffer inside the registered slice and
    # leaves the paired course poles untouched.
    assert torch.equal(model_batch.context, context_before)
    assert torch.allclose(
        algorithm.model.advance_context(
            model_batch.context, model_batch.actions[:, 0]
        )[:, columns],
        expected,
    )

    policy_batch = replay.sample(batch_size=8, window=1, history=10)
    assert torch.allclose(policy_batch.context[:, columns], expected)
    actor_before = _parameters(actor)
    critic_before = _parameters(critic)
    warmup_metrics = algorithm.improve_policy(
        policy_batch,
        critic_epochs=1,
        critic_minibatches=1,
        update_actor=False,
    )
    assert torch.equal(actor_before, _parameters(actor))
    assert not torch.equal(critic_before, _parameters(critic))
    assert warmup_metrics["actor_grad_norm"] == 0.0
    assert np.isfinite(warmup_metrics["critic_loss"])

    actor_before = _parameters(actor)
    critic_before = _parameters(critic)
    metrics = algorithm.improve_policy(
        policy_batch, critic_epochs=1, critic_minibatches=1
    )
    assert {
        "actor_loss",
        "actor_grad_norm",
        "critic_loss",
        "imagined_reward",
        "imagination_survival_fraction",
        "critic_target_abs_max",
    } == set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(actor_before, _parameters(actor))
    assert not torch.equal(critic_before, _parameters(critic))
    assert torch.allclose(policy_batch.context[:, columns], expected)
    assert torch.allclose(
        algorithm.model.advance_context(
            policy_batch.context, policy_batch.actions[:, 0]
        )[:, columns],
        expected,
    )
