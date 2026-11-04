"""Source-profile collection, temporal replay, imagination, and evaluation.

Every component takes the paired ``CourseCase`` as the unit of identity.  A
rollout records the case hash, its clip index in the motion library, and the
reference frame of each transition; replay stores that frame and the 149-value
course context; imagination rebuilds the reference window from the same clip
and frame; evaluation reloads the same registered pair.  Nothing here consumes
soccer ball or goal state.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from supermimic.algorithm import SuperMimic, gae_returns
from supermimic.imagination_terminal import absorbing_inputs, surviving_reward_mean
from supermimic.models import frozen
from supermimic.randomization import nominal_context
from supermimic.replay import ReplayBatch, TemporalReplay
from supermimic.reward_tuning import RewardTuningCfg, tuned_reward

from .data import CourseCase, PoleMotions
from .source_environment import SourcePoleEnvironment
from .source_profile import (
    SOURCE_ACTION_DIM,
    SOURCE_CONTEXT_DIM,
    SOURCE_HISTORY_FRAMES,
    SOURCE_OBS_DIM,
    SOURCE_STATE_DIM,
    SourcePoleRobot,
    source_goal_features,
    validate_source_history,
)

# Source task 7.7 reward tuning: only the eight arm joints are penalized.
SOURCE_REWARD_TUNING = RewardTuningCfg(arm_joint_weight=0.05, action_scale=0.5)


def source_nominal_context(robot: SourcePoleRobot, poles_xy: Tensor) -> Tensor:
    """Return the source nominal context with the paired pole columns set."""
    if poles_xy.shape[-2:] != (3, 2):
        raise ValueError("Course poles must have shape (..., 3, 2)")
    context = nominal_context(
        robot.context_layout, int(np.prod(poles_xy.shape[:-2])), poles_xy.device
    )
    return robot.set_course_poles(context, poles_xy)


def advance_source_context(
    robot: SourcePoleRobot, context: Tensor, executed_action: Tensor
) -> Tensor:
    """Advance the delay buffer without overwriting the course-pole columns.

    The vendored world model appends an action to the last 22 columns, which in
    the 149-column source layout are the six pole coordinates and the tail of
    the delay buffer.  The profile layout is authoritative, so write the
    source's ``previous_executed_action`` slice and keep the paired course.
    """
    if executed_action.shape[-1] != SOURCE_ACTION_DIM:
        raise ValueError("Executed source action must have 22 values")
    advanced = context.clone()
    advanced[..., robot.context_layout.previous_executed_action] = executed_action
    return advanced


def _reference_window(
    motions: PoleMotions, clip: int, frame: int, offsets: Tensor
) -> Tensor:
    clips = torch.tensor([clip], dtype=torch.long, device=offsets.device)
    frames = torch.tensor([frame], dtype=torch.long, device=offsets.device)
    return motions.sequence(clips, frames, offsets)


@dataclass(frozen=True)
class SourceEpisode:
    """One real-simulation rollout with its paired reference/course identity."""

    case: CourseCase
    clip: int
    start_frame: int
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    boundaries: np.ndarray
    contexts: np.ndarray
    frames: np.ndarray
    transitions: tuple[dict, ...]
    result: dict

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def identity(self) -> str:
        return self.case.identity


@contextmanager
def _optional_viewer(environment: SourcePoleEnvironment, show_gui: bool):
    """Yield a passive viewer only when playback explicitly asks for one."""
    if not show_gui:
        yield None
        return
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(environment.model, environment.data)
    try:
        poles_xy = np.asarray(environment.case.poles_xy, dtype=np.float64)
        with viewer.lock():
            viewer.cam.lookat[:] = [
                float((poles_xy[:, 0].min() + poles_xy[:, 0].max()) / 2),
                float((poles_xy[:, 1].min() + poles_xy[:, 1].max()) / 2),
                0.65,
            ]
            viewer.cam.distance = max(4.5, float(np.ptp(poles_xy[:, 0])) * 1.35)
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -45
        yield viewer
    finally:
        viewer.close()


@torch.no_grad()
def run_source_episode(
    actor: Callable[[Tensor, Tensor], Tensor],
    robot: SourcePoleRobot,
    motions: PoleMotions,
    case: CourseCase,
    clip: int,
    scene_path,
    *,
    start_frame: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    action_noise: float = 0.0,
    observation_noise: bool = False,
    clean_policy_actions: bool = True,
    reward_tuning: RewardTuningCfg = SOURCE_REWARD_TUNING,
    max_steps: int | None = None,
    randomize: bool = False,
    show_gui: bool = False,
) -> SourceEpisode:
    """Roll out one paired course in the source profile.

    ``start_frame`` defaults to the source training sampler: a uniform frame in
    ``1 .. len(reference) - 2``.  Evaluation passes ``start_frame=0``.
    """
    if not 0 <= clip < len(motions.cases):
        raise ValueError("Clip index is outside the paired motion library")
    if motions.cases[clip].identity != case.identity:
        raise ValueError("Motion clip and rollout case are not the same paired reference")
    environment = SourcePoleEnvironment(
        case, scene_path, robot, seed=seed, randomize=randomize
    )
    rng = np.random.default_rng(seed)
    last_frame = len(case.qpos) - 1
    if start_frame is None:
        start_frame = int(np.floor(rng.random() * max(1, last_frame)))
        start_frame = min(max(start_frame, 1), max(1, last_frame - 1))
    if not 0 <= start_frame <= last_frame:
        raise ValueError(f"Start frame {start_frame} is outside {case.case_id}")

    state_np = environment.reset(start_frame)
    frame = start_frame
    poles = torch.as_tensor(case.poles_xy, dtype=torch.float32, device=device).unsqueeze(0)
    context = (
        environment.context.to(device)
        if randomize
        else source_nominal_context(robot, poles)
    )
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    observation = robot.observation(state, context)
    if observation_noise:
        observation = observation + torch.randn_like(observation) * robot.noise_scale()
    history = observation[:, None].expand(-1, SOURCE_HISTORY_FRAMES, -1).clone()

    states, observations, actions = [state_np.copy()], [observation[0].cpu().numpy().copy()], []
    rewards, terminated_flags, boundaries, contexts = [], [], [], []
    frames = []
    transitions: list[dict] = []
    fall = False
    done = False
    step = 0
    with _optional_viewer(environment, show_gui) as viewer:
        while not done:
            validate_source_history(history)
            references = _reference_window(
                motions, clip, frame, torch.arange(1, 11, device=device)
            )
            goal = source_goal_features(robot, references, state)
            action = actor(history, goal)
            if action.shape[-1] != SOURCE_ACTION_DIM:
                raise ValueError("Source actor must output 22 actions")
            executed = action
            if action_noise:
                executed = action + rng.normal(0.0, action_noise, 22).astype(np.float32)
            reward_action = action if clean_policy_actions else executed
            replay_context = context.clone()
            next_state_np, transition = environment.step(executed[0].cpu().numpy())
            next_state = torch.as_tensor(
                next_state_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            next_frame = min(frame + 1, last_frame)
            target = references[:, 0]
            reward_state = state
            if clean_policy_actions:
                previous_action = robot.previous_action_from_observation(history[..., -1, :])
                reward_state = torch.cat((state[..., :57], previous_action), -1)
            reward = tuned_reward(
                robot,
                reward_state,
                next_state,
                target,
                reward_action,
                motions.dt,
                tuning=reward_tuning,
                context=replay_context,
            )
            fall_now = bool(robot.terminated(next_state, target)[0])
            fall = fall or fall_now
            pole_collision = bool(transition["pole_collision"])
            terminal = fall_now or pole_collision
            # Pushes are exogenous: source replay starts a fresh sampling segment
            # from the perturbed state so no imagined window crosses the impulse.
            boundary = terminal or next_frame >= last_frame or bool(transition["pushed"])
            cause = (
                "pole_collision"
                if pole_collision
                else "fall"
                if fall_now
                else "route_end"
                if next_frame >= last_frame
                else None
            )

            frames.append(frame)
            contexts.append(replay_context[0].cpu().numpy().copy())
            actions.append(executed[0].cpu().numpy().copy())
            rewards.append(float(reward[0]))
            terminated_flags.append(float(terminal))
            boundaries.append(bool(boundary))
            states.append(next_state_np.copy())
            transition = {
                **transition,
                "context": replay_context[0].cpu().numpy().copy(),
                "observation": history[0, -1].cpu().numpy().copy(),
                "frame": frame,
                "clip": clip,
                "reward": float(reward[0]),
                "terminated": float(terminal),
                "boundary": bool(boundary),
                "fall": fall_now,
                "terminal_cause": cause,
            }
            transitions.append(transition)

            observation = robot.observation(next_state, context)
            if observation_noise:
                observation = observation + torch.randn_like(observation) * robot.noise_scale()
            observations.append(observation[0].cpu().numpy().copy())
            history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
            context = advance_source_context(robot, context, executed)
            state = next_state
            frame = next_frame
            step += 1
            done = boundary or (max_steps is not None and step >= max_steps)
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    done = True

    if transitions and not boundaries[-1]:
        # A caller-bounded truncation is still an episode boundary: no replay
        # window may splice it into the next episode.
        transitions[-1]["boundary"] = True
        boundaries[-1] = True

    result = _episode_result(
        case, environment, states, transitions, steps=len(actions), fall=fall
    )
    return SourceEpisode(
        case=case,
        clip=clip,
        start_frame=start_frame,
        states=np.asarray(states, dtype=np.float32),
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.float32),
        boundaries=np.asarray(boundaries, dtype=bool),
        contexts=np.asarray(contexts, dtype=np.float32),
        frames=np.asarray(frames, dtype=np.int64),
        transitions=tuple(transitions),
        result=result,
    )


def _episode_result(case, environment, states, transitions, *, steps, fall):
    roots = np.asarray([state[:2] for state in states], dtype=np.float64)
    targets = np.asarray(
        [case.qpos[min(frame + 1, len(case.qpos) - 1), :2] for frame in (
            transition["frame"] for transition in transitions
        )],
        dtype=np.float64,
    )
    errors = roots[1:] - targets
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf")
    endpoint = float(np.linalg.norm(roots[-1] - case.qpos[-1, :2]))
    collision = bool(environment.collision)
    route_completed = environment.frame == len(case.qpos) - 1 and endpoint <= 0.5
    contacts = sorted({name for row in transitions for name in row["pole_contacts"]})
    return {
        "case_id": case.case_id,
        "reference_id": case.identity,
        "steps": steps,
        "endpoint_distance_m": endpoint,
        "route_completed": route_completed,
        "pole_collision": collision,
        "pole_contacts": contacts,
        "fall": bool(fall),
        "root_xy_rmse_m": rmse,
        "success": bool(
            route_completed and not collision and not fall and rmse <= 0.5
        ),
    }


def make_source_replay(
    episodes: list[SourceEpisode],
    robot: SourcePoleRobot,
    clip_by_case: dict[str, int],
    *,
    device: str = "cpu",
    capacity: int | None = None,
) -> TemporalReplay:
    """Build source-order replay rows, refusing any broken course pairing."""
    total = sum(episode.steps for episode in episodes)
    if total == 0:
        raise ValueError("Source replay needs at least one transition")
    replay = TemporalReplay(
        capacity=max(int(capacity or 256), total),
        num_envs=1,
        state_dim=SOURCE_STATE_DIM,
        context_dim=robot.context_dim,
        obs_dim=SOURCE_OBS_DIM,
        device=device,
    )
    for episode in episodes:
        if episode.clip != clip_by_case[episode.case.case_id]:
            raise ValueError("Episode clip differs from its registered case index")
        poles = episode.case.poles_xy.reshape(-1)
        for transition in episode.transitions:
            if transition["reference_id"] != episode.case.identity:
                raise ValueError("Transition reference identity differs from its episode")
            if not np.array_equal(np.asarray(transition["pole_xy"]), episode.case.poles_xy):
                raise ValueError("Transition pole layout differs from its paired course")
            stored_poles = np.asarray(transition["context"])[
                robot.context_layout.course_poles
            ]
            if not np.allclose(stored_poles, poles):
                raise ValueError("Transition context does not carry its paired course poles")
            add = lambda value, dtype=torch.float32: torch.as_tensor(
                np.asarray(value, dtype=np.float32), dtype=dtype, device=device
            ).reshape(1, -1)
            replay.add(
                states=add(transition["state"]),
                next_states=add(transition["next_state"]),
                actions=add(transition["action"]),
                context=add(transition["context"]),
                observation=add(transition["observation"]),
                terminated=torch.tensor([transition["terminated"]], device=device),
                boundary=torch.tensor([transition["boundary"]], dtype=torch.bool, device=device),
                clips=torch.tensor([episode.clip], dtype=torch.long, device=device),
                frames=torch.tensor([transition["frame"]], dtype=torch.long, device=device),
            )
    return replay


def source_batch_references(
    motions: PoleMotions, batch: ReplayBatch, offsets: Tensor
) -> Tensor:
    """Gather references by the batch's paired clip/frame identity."""
    return motions.sequence(batch.clips, batch.frames, offsets)


def source_batch_goals(
    robot: SourcePoleRobot, motions: PoleMotions, batch: ReplayBatch
) -> Tensor:
    """Rebuild the ten-frame robot-anchored goals for a sampled batch."""
    references = source_batch_references(
        motions, batch, torch.arange(1, 11, device=batch.states.device)
    )
    _check_batch_course(robot, motions, batch)
    return source_goal_features(robot, references, batch.states[:, 0])


def _check_batch_course(
    robot: SourcePoleRobot, motions: PoleMotions, batch: ReplayBatch
) -> None:
    if batch.clips.max().item() >= len(motions.cases):
        raise ValueError("Replay clip index is outside the paired motion library")
    expected = motions.poles_xy.to(batch.context.device)[batch.clips]
    stored = batch.context[:, robot.context_layout.course_poles].reshape(-1, 3, 2)
    if not torch.allclose(stored, expected):
        raise ValueError("Sampled batch context does not match its paired course")


class SourcePoleSuperMimic(SuperMimic):
    """SuperMimic core wired to the source robot-anchored goal and context.

    The pinned vendor revision predates the source task's robot-anchored goal
    mode: its imagination uses a reference-only goal and its clean-action
    slice assumes the previous action is the last 22 observation columns.  The
    source profile supplies both semantics explicitly without changing the
    vendor pin or the soccer profile.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("cache_goals", False):
            raise ValueError("Source robot-anchored goals depend on live state and cannot be cached")
        super().__init__(*args, **kwargs)

    def _reference_goals(self, references: Tensor) -> Tensor:
        raise RuntimeError("Source robot-anchored goals cannot be cached")

    def _imagined_step(
        self,
        state,
        history,
        references,
        context,
        target_heads=None,
        cached_goal=None,
        encoding=None,
        active=None,
    ):
        goal = (
            source_goal_features(self.robot, references, state)
            if cached_goal is None
            else cached_goal
        )
        value = self.critic(state, goal, context)
        action = (
            self.actor(history, goal)
            if encoding is None
            else self.actor.decode(history, encoding)
        )
        next_state = self.model(state, action, context)
        if self.clean_policy_actions:
            previous_action = self.robot.previous_action_from_observation(history[..., -1, :])
            reward_state = torch.cat((state[..., :57], previous_action), -1)
        else:
            reward_state = state
        reward = tuned_reward(
            self.robot,
            reward_state,
            next_state,
            references[:, 0],
            action,
            self.motions.dt,
            tuning=self.reward_tuning,
            context=context,
            target_heads=target_heads,
        )
        termination = self.model.termination_logits(
            self.robot, next_state, references[:, 0]
        ).sigmoid()
        observation = self.robot.observation(next_state, context)
        if active is not None:
            reward = torch.where(active, reward, 0)
            termination = torch.where(active, termination, 1)
            value = torch.where(active, value, 0)
        return next_state, reward, termination, observation, goal, value

    def improve_policy(
        self,
        batch: ReplayBatch,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        *,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """Vendored update flow with the source robot-anchored bootstrap goal.

        The warmup stage calls this with ``update_actor=False``: the actor is
        frozen, but the world model remains fixed and the critic still fits the
        imagined returns, matching source task 7.7's stage transition.
        """
        if update_actor and self.imagination_graph is not None and batch.states.is_cuda:
            raise RuntimeError("Source profile does not support captured imagination")
        h = self.horizon
        references = source_batch_references(
            self.motions, batch, torch.arange(h + 11, device=batch.frames.device)
        )
        remaining = self.motions.remaining(batch.clips, batch.frames)
        reference_heads = None
        if self.cache_reference_heads:
            with torch.no_grad():
                flat = references[:, 1 : h + 1].reshape(-1, self.robot.state_dim)
                reference_heads = tuple(
                    head.reshape(len(references), h, *head.shape[1:])
                    for head in self.reference_kinematics(flat)
                )
        state, history = batch.states[:, 0].detach(), batch.history.detach()
        context = batch.context
        initial_state, initial_history, initial_context = state, history, context
        survival = state.new_ones(len(state))
        objective = state.new_zeros(len(state))
        states, goals, rewards, terminations, values = [], [], [], [], []
        contexts, weights = [], []
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen(self.model, self.critic), (
            nullcontext() if update_actor else frozen(self.actor)
        ):
            for t in range(h):
                state, history, context, active = absorbing_inputs(
                    survival,
                    state,
                    history,
                    context,
                    initial_state,
                    initial_history,
                    initial_context,
                )
                weights.append(torch.where(active, survival.detach(), 0))
                next_state, reward, termination, observation, goal, value = (
                    self.imagined_step(
                        state,
                        history,
                        references[:, t + 1 : t + 11].contiguous(),
                        context,
                        (
                            reference_heads[0][:, t].contiguous(),
                            reference_heads[1][:, t].contiguous(),
                            reference_heads[2][:, t].contiguous(),
                            reference_heads[3][:, t].contiguous(),
                        )
                        if reference_heads is not None
                        else None,
                        None,
                        None,
                        active,
                    )
                )
                states.append(state.detach())
                contexts.append(context.detach())
                goals.append(goal.detach())
                values.append(value)
                termination = torch.where(
                    remaining <= t + 1, torch.ones_like(termination), termination
                )
                objective = objective + self.gamma**t * survival * reward
                survival = survival * (1 - termination)
                rewards.append(reward)
                terminations.append(termination)
                observation = (
                    observation + torch.randn_like(observation) * self.robot.noise_scale()
                )
                history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
                state = next_state
                context = advance_source_context(
                    self.robot, context, next_state[:, 57:79]
                )
            final_goal = source_goal_features(
                self.robot, references[:, h + 1 : h + 11], state
            )
            state, history, context, active = absorbing_inputs(
                survival,
                state,
                history,
                context,
                initial_state,
                initial_history,
                initial_context,
            )
            survival = torch.where(active, survival, 0)
            values.append(torch.where(active, self.critic(state, final_goal, context), 0))
            objective = objective + self.gamma**h * survival * values[-1]
            loss = -objective.mean()
            if self.normalize_policy_objective:
                loss = loss / h
            self.distributed.require_finite(loss, "Nonfinite imagined return")
            if update_actor:
                loss.backward()
        actor_norm = self._update_actor() if update_actor else loss.new_zeros(())
        returns = gae_returns(
            torch.stack(rewards, 1), torch.stack(terminations, 1), torch.stack(values, 1)
        )
        x = torch.stack(states, 1).flatten(0, 1)
        g = torch.stack(goals, 1).flatten(0, 1)
        ctx = torch.stack(contexts, 1).flatten(0, 1)
        weights_tensor = torch.stack(weights, 1)
        critic_loss = self._fit_critic(
            x,
            g,
            ctx,
            returns.flatten(),
            weights_tensor.flatten(),
            critic_epochs,
            critic_minibatches,
        )
        return {
            "actor_loss": float(loss.detach()),
            "actor_grad_norm": float(actor_norm),
            "critic_loss": float(critic_loss),
            "imagined_reward": float(
                surviving_reward_mean(torch.stack(rewards, 1).detach(), weights_tensor)
            ),
            "imagination_survival_fraction": float(weights_tensor.mean()),
            "critic_target_abs_max": float(returns.abs().max()),
        }


@torch.no_grad()
def evaluate_source_cases(
    actor: Callable[[Tensor, Tensor], Tensor],
    robot: SourcePoleRobot,
    motions: PoleMotions,
    cases: list[CourseCase],
    scene_path,
    *,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, dict]:
    """Fixed-seed headless evaluation of each registered course/reference pair."""
    by_id = {case.case_id: index for index, case in enumerate(motions.cases)}
    reports = {}
    for index, case in enumerate(cases):
        if case.case_id not in by_id:
            raise ValueError(f"Evaluation case {case.case_id} is absent from the motion library")
        if motions.cases[by_id[case.case_id]].identity != case.identity:
            raise ValueError(f"Evaluation case {case.case_id} is not the paired reference")
        episode = run_source_episode(
            actor,
            robot,
            motions,
            case,
            by_id[case.case_id],
            scene_path,
            start_frame=0,
            seed=seed + index,
            device=device,
            observation_noise=False,
            action_noise=0.0,
        )
        reports[case.case_id] = episode.result
    return reports


__all__ = [
    "SOURCE_REWARD_TUNING",
    "SourceEpisode",
    "SourcePoleSuperMimic",
    "advance_source_context",
    "evaluate_source_cases",
    "make_source_replay",
    "run_source_episode",
    "source_batch_goals",
    "source_batch_references",
    "source_nominal_context",
]
