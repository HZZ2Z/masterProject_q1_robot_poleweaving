"""Pole-task adapters over the project's expert policy and SuperMimic core."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from expert.policy import ActorCritic
from supermimic.algorithm import SuperMimic
from supermimic.models import Critic, MotionPolicy, WorldModel, goal_features
from supermimic.robot import Q1Robot, normalize_quat, quat_apply, quat_inverse, quat_mul

from .data import policy_course_features

FEATURE_DIM = 276
CRITIC_GOAL_DIM = 576
GOAL_DIM = CRITIC_GOAL_DIM + 7
ROUTE_CONDITION_DIM = 7
POLE_CONTEXT_DIM = 6
POLE_RADIUS_M = 0.04
POLE_SAFE_CLEARANCE_M = 0.35

SOURCE_ACTOR_HIDDEN = (2048, 1024, 512, 256)
SOURCE_MODEL_HIDDEN = (2048, 2048, 2048)
SOURCE_CRITIC_HIDDEN = (256, 128)


class SourcePoleActor(MotionPolicy):
    """Vendor motion policy with the source pole profile's 78-value frame."""

    def __init__(self, robot):
        if robot.obs_dim != 78:
            raise ValueError(f"Source pole actor requires 78 observation values, got {robot.obs_dim}")
        super().__init__(robot, hidden=SOURCE_ACTOR_HIDDEN, activation="silu")
        self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(6)))


class SourcePoleWorldModel(WorldModel):
    """Vendor world model configured to the source task's heading contract."""

    def __init__(self, robot):
        if robot.state_dim != 79 or robot.context_dim != 149:
            raise ValueError("Source pole world model requires 79-state / 149-context input")
        super().__init__(
            robot,
            hidden=SOURCE_MODEL_HIDDEN,
            activation="silu",
            representation="heading",
        )
        # The vendored default appends the action to the last 22 columns, but
        # this profile's 149-column layout ends with the six course-pole
        # coordinates.  Keep the delay buffer in its registered slice so
        # model fitting and imagination never overwrite the paired course.
        self.previous_action_slice = getattr(
            robot.context_layout,
            "previous_executed_action",
            slice(-22, None),
        )

    def advance_context(self, context, executed_action):
        if not self.modeled_action_delay:
            return context
        if executed_action.shape[-1] != 22:
            raise ValueError("Source executed action must have 22 values")
        advanced = context.clone()
        advanced[..., self.previous_action_slice] = executed_action
        return advanced


class SourcePoleCritic(Critic):
    """Vendor critic configured to the source task's paired-course input."""

    def __init__(self, robot):
        if robot.state_dim != 79 or robot.physical_dim != 57 or robot.context_dim != 149:
            raise ValueError("Source pole critic requires 79-state / 57-goal / 149-context input")
        super().__init__(robot, hidden=SOURCE_CRITIC_HIDDEN, activation="silu")


def build_source_models(robot):
    """Construct vendor model classes with the source checkpoint's exact shape contract."""
    return SourcePoleActor(robot), SourcePoleWorldModel(robot), SourcePoleCritic(robot)


class PoleActor(nn.Module):
    """Project expert actor with zero-start route residual and vendor API."""

    goal_embedding_dim = POLE_CONTEXT_DIM + ROUTE_CONDITION_DIM

    def __init__(self, expert: ActorCritic):
        super().__init__()
        self.encoder = copy.deepcopy(expert.encoder)
        self.actor = copy.deepcopy(expert.actor)
        for module in (self.encoder, self.actor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.route_residual = nn.Sequential(
            nn.Linear(79 + POLE_CONTEXT_DIM + ROUTE_CONDITION_DIM, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 22),
        )
        nn.init.zeros_(self.route_residual[-1].weight)
        nn.init.zeros_(self.route_residual[-1].bias)

    def _expert_action(self, observation: Tensor, newest_first: Tensor) -> Tensor:
        embedding = self.encoder(newest_first.flatten(start_dim=-2))
        return self.actor(torch.cat((observation, embedding), -1))

    def forward_control(self, observation: Tensor, history: Tensor, goal: Tensor) -> Tensor:
        if goal.shape[-1] != GOAL_DIM:
            raise ValueError(f"Pole control goal must have {GOAL_DIM} features")
        base = self._expert_action(observation, history)
        return (base + self._route_residual(self.encode_goal(goal), observation)).clamp(-1, 1)

    def encode_goal(self, goal: Tensor) -> Tensor:
        if goal.shape[-1] != GOAL_DIM:
            raise ValueError(f"SuperMimic pole goal must have {GOAL_DIM} features")
        encoded = goal[..., -self.goal_embedding_dim:]
        no_course_poles = encoded[..., :POLE_CONTEXT_DIM].abs().amax(-1) >= 50.0
        pole_features = torch.where(
            no_course_poles[..., None],
            torch.zeros_like(encoded[..., :POLE_CONTEXT_DIM]),
            encoded[..., :POLE_CONTEXT_DIM],
        )
        return torch.cat((pole_features, encoded[..., POLE_CONTEXT_DIM:]), -1)

    def _route_residual(self, encoding: Tensor, observation: Tensor) -> Tensor:
        residual = self.route_residual(torch.cat((observation, encoding), -1)).tanh()
        return 0.5 * residual

    def decode(self, history: Tensor, encoding: Tensor) -> Tensor:
        if history.shape[-2:] != (50, 79):
            raise ValueError("Pole expert history must be 50 frames by 79 observations")
        observation = history[:, 0]
        return (self._expert_action(observation, history) + self._route_residual(encoding, observation)).clamp(-1, 1)

    def forward(self, *args: Tensor) -> Tensor:
        if len(args) == 3:
            return self.forward_control(args[0], args[1], args[2])
        if len(args) == 2:
            history, goal = args
            return self.decode(history, self.encode_goal(goal))
        raise TypeError("PoleActor expects (history, goal) or (observation, history, goal)")

    @classmethod
    def from_project_expert(cls, checkpoint_path: Path, expert_cfg: dict) -> tuple["PoleActor", dict]:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
            raise ValueError("Expected project Q1 expert checkpoint with a model state dictionary")
        expert = ActorCritic(expert_cfg)
        expert.load_state_dict(saved["model"], strict=True)
        actor = cls(expert)
        observation = torch.zeros((2, expert_cfg["env"]["num_observations"]))
        history = torch.zeros((2, expert_cfg["runner"]["num_stack"], expert_cfg["env"]["num_observations"]))
        features = torch.randn((2, GOAL_DIM))
        with torch.no_grad():
            parity = (expert(observation, history) - actor.forward_control(observation, history, features)).abs().max().item()
        if parity > 1e-6:
            raise ValueError(f"Pole actor warm-start action parity failed: max error {parity:.3g}")
        return actor, {
            "source": str(checkpoint_path.resolve()),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "encoder": "expert.encoder -> pole.encoder (exact copy)",
            "action_head": "expert.actor -> pole.actor (exact copy)",
            "trainable_scope": "freeze expert encoder/action head; train only the zero-initialized route residual adapter",
            "route_residual": "zero-initialized (79 observation + 6 local pole + 7 route features)-to-(128, 128, 22), tanh scaled by 0.5",
            "goal_encoder": "vendor reference goal remains frozen; route adapter receives ordered live robot-frame poles and 7 route features",
            "course_features": FEATURE_DIM,
            "vendor_goal_features": GOAL_DIM,
            "actions": 22,
            "action_parity_max_abs": parity,
            "action_parity_tolerance": 1e-6,
        }


class PoleRobot(Q1Robot):
    """Q1 vendor robot contract with pole context and expert observations."""

    def __init__(self, xml: Path):
        super().__init__(xml)
        self.base_context_dim = self.context_dim
        self.context_dim += POLE_CONTEXT_DIM
        self.obs_dim = 79

    def context_for_poles(self, poles_xy: Tensor, context: Tensor | None = None) -> Tensor:
        if poles_xy.shape[-2:] != (3, 2):
            raise ValueError("Pole context requires three ordered XY pole centers")
        if context is None:
            context = poles_xy.new_zeros((*poles_xy.shape[:-2], self.base_context_dim))
        if context.shape[-1] != self.base_context_dim:
            raise ValueError("Base Q1 context dimension mismatch")
        has_poles = poles_xy[..., 0, 0] < 50.0
        encoded_poles = torch.where(has_poles[..., None, None], poles_xy, torch.zeros_like(poles_xy))
        return torch.cat((context, encoded_poles.flatten(start_dim=-2)), -1)

    def observation(self, state: Tensor, context: Tensor) -> Tensor:
        return observation_from_state(state, self.default_q)

    def noise_scale(self) -> Tensor:
        return self.default_q.new_zeros(79)

    def reward(self, state, next_state, target, action, dt, target_heads=None):
        base = super().reward(state, next_state, target, action, dt, target_heads)
        poles = target[..., 79:85].reshape(*target.shape[:-1], 3, 2)
        body_positions = self.kinematics(next_state)[0][..., 1:, :2]
        clearance = (body_positions[..., :, None, :] - poles[..., None, :, :]).norm(dim=-1).amin(dim=(-1, -2))
        clearance = clearance - POLE_RADIUS_M - 0.12
        deficit = (POLE_SAFE_CLEARANCE_M - clearance).clamp_min(0).square()
        has_poles = poles[..., 0, 0] < 50.0
        deficit = torch.where(has_poles, deficit, torch.zeros_like(deficit))
        smoothing = (action - state[..., 57:79]).square().mean(-1)
        return base - dt * (0.25 * deficit + 0.05 * smoothing)


class PoleWorldModel(WorldModel):
    """Vendor world model conditioned on the paired course context."""

    def __init__(self, robot: PoleRobot, hidden=(512, 512, 512)):
        super().__init__(robot, tuple(hidden), representation="legacy")


class PoleCritic(Critic):
    """Vendor critic extended by six ordered robot-frame pole features."""

    def __init__(self, robot: PoleRobot, hidden=(512, 256)):
        super().__init__(robot, tuple(hidden))
        first = self.network[0]
        assert isinstance(first, nn.Linear)
        widened = nn.Linear(first.in_features + POLE_CONTEXT_DIM, first.out_features)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, : first.in_features].copy_(first.weight)
            widened.bias.copy_(first.bias)
        self.network[0] = widened

    def forward(self, state: Tensor, goal: Tensor, context: Tensor) -> Tensor:
        if goal.shape[-1] == GOAL_DIM:
            goal = goal[..., :CRITIC_GOAL_DIM]
        if goal.shape[-1] == CRITIC_GOAL_DIM - POLE_CONTEXT_DIM:
            poles = context[..., -POLE_CONTEXT_DIM:].reshape(-1, 3, 2)
            delta = poles - state[:, None, :2]
            yaw = yaw_from_state(state)
            c, s = yaw.cos()[:, None], yaw.sin()[:, None]
            poles_local = torch.stack(
                (c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]),
                -1,
            ).flatten(1)
            goal = torch.cat((goal, poles_local), -1)
        if goal.shape[-1] != CRITIC_GOAL_DIM:
            raise ValueError(f"Pole critic goal must have {CRITIC_GOAL_DIM} features")
        return super().forward(state, goal, context)


class PoleSuperMimic(SuperMimic):
    """Vendor learner with pole-conditioned reference goals and reward hooks."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("cache_goals", False):
            raise ValueError("Pole goals depend on live robot state; cache_goals is unsupported")
        super().__init__(*args, **kwargs)

    def _with_pole_goal(self, references: Tensor, state: Tensor) -> Tensor:
        return pole_actor_goal_features(self.robot, references, state)

    def _reference_goals(self, references: Tensor) -> Tensor:
        raise RuntimeError("Pole goals depend on live robot state and cannot be cached")

    def _imagined_step(self, state, history, references, context, target_heads=None, cached_goal=None, encoding=None, active=None):
        if cached_goal is not None or encoding is not None:
            raise ValueError("Pole goal caching must include the live robot state")
        cached_goal = self._with_pole_goal(references, state)
        encoding = self.actor.encode_goal(cached_goal)
        return super()._imagined_step(
            state, history, references, context, target_heads,
            cached_goal=cached_goal, encoding=encoding, active=active,
        )


def pole_goal_features(robot: PoleRobot, references: Tensor, state: Tensor) -> Tensor:
    """Build the shared vendor goal plus ordered robot-frame pole coordinates."""
    if references.shape[1] < 10 or references.shape[-1] != 85:
        raise ValueError("Pole goals require ten paired 85-value reference states")
    if state.shape[-1] < 7 or state.shape[0] != references.shape[0]:
        raise ValueError("Pole goals require a matching batch of live robot states")
    base = goal_features(robot, references)
    poles = references[:, 0, 79:85].reshape(-1, 3, 2)
    encoded = policy_course_features(
        references[:, :10, :29], poles, state[:, :2], yaw_from_state(state)
    )
    return torch.cat((base, encoded[:, -POLE_CONTEXT_DIM:]), -1)


def pole_actor_goal_features(robot: PoleRobot, references: Tensor, state: Tensor) -> Tensor:
    """Append live route progress and steering features for the actor adapter."""
    return torch.cat(
        (pole_goal_features(robot, references, state), route_condition(robot, references, state)),
        -1,
    )


def route_condition(robot: PoleRobot, references: Tensor, state: Tensor) -> Tensor:
    """Encode the next pole, route progress, and current reference tracking error."""
    poles = references[:, 0, 79:85].reshape(-1, 3, 2)
    has_poles = poles[:, 0, 0] < 50.0
    route_poles = poles - poles[:, :1]
    route_x = state[:, 0] - poles[:, 0, 0]
    active = route_poles[..., 0] < 500.0
    ahead = (route_poles[..., 0] >= route_x[:, None] - 0.25) & active
    distance = torch.where(ahead, route_poles[..., 0] - route_x[:, None], torch.inf)
    next_index = distance.argmin(-1)
    last_index = active.sum(-1).clamp_min(1) - 1
    next_index = torch.where(ahead.any(-1), next_index, last_index)
    one_hot = torch.nn.functional.one_hot(next_index, 3).to(state.dtype)
    side = state.new_tensor([1.0, -1.0, 1.0])[next_index]
    last_x = route_poles[..., 0].gather(-1, last_index[:, None]).squeeze(-1)
    progress = ((route_x + 1.2) / (last_x + 2.4).clamp_min(1e-6)).clamp(0, 1)
    lateral = (references[:, 0, 1] - state[:, 1]).clamp(-1, 1)
    relative = quat_mul(
        quat_inverse(normalize_quat(state[:, 3:7])),
        normalize_quat(references[:, 0, 3:7]),
    )
    heading = (2 * torch.atan2(relative[:, 3], relative[:, 0]) / torch.pi).clamp(-1, 1)
    features = torch.cat(
        (one_hot, side[:, None], progress[:, None], lateral[:, None], heading[:, None]), -1
    )
    no_pole_features = torch.cat((features.new_zeros((len(features), 5)), features[:, 5:]), -1)
    return torch.where(has_poles[:, None], features, no_pole_features)


def observation_from_state(state: Tensor, default_q: Tensor | None = None) -> Tensor:
    """Construct the project's 79-value actor observation from vendor state."""
    q = normalize_quat(state[..., 3:7])
    gravity = quat_apply(quat_inverse(q), state.new_tensor([0.0, 0.0, -1.0]))
    gyro = quat_apply(quat_inverse(q), state[..., 32:35])
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), w.square() + x.square() - y.square() - z.square())
    if default_q is None:
        default_q = state.new_zeros(22)
    return torch.cat(
        (
            gravity, gyro, state.new_zeros((*state.shape[:-1], 5)),
            yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1),
            state[..., 7:29] - default_q, state[..., 35:57] * 0.1, state[..., 57:79],
        ), -1,
    )


def yaw_from_state(state: Tensor) -> Tensor:
    q = normalize_quat(state[..., 3:7])
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), w.square() + x.square() - y.square() - z.square())


def features_for_frames(case, frames: Tensor, state: Tensor) -> Tensor:
    """Build paired future-reference/pole features for a vector of frames."""
    frame_offsets = torch.arange(1, 11, device=frames.device)
    indexes = (frames[:, None] + frame_offsets[None, :]).clamp_max(len(case.qpos) - 1)
    reference = torch.as_tensor(case.qpos, dtype=state.dtype, device=state.device)[indexes]
    poles = torch.as_tensor(case.poles_xy, dtype=state.dtype, device=state.device)
    poles = poles.unsqueeze(0).expand(len(state), -1, -1)
    return policy_course_features(reference, poles, state[:, :2], yaw_from_state(state))
