"""Source task 7.7 pole profile: robot, observation, goal, and action contract.

The registered warmup checkpoint was trained against the source project's
course-conditioned ``Q1Robot`` semantics.  This module reproduces those
semantics on top of the pinned vendored SuperMimic revision without adding the
later ``speed_encoder``/``route_encoder``/``route_adapter`` modules that are
absent from that checkpoint.

Tensor contracts (also recorded in ``pole_task/source_manifest.json``):

* state                79 = qpos 29 + qvel 28 + previous action 22
* per-frame observation 78 = joints 22 + joint velocities 22 + gyro 3 +
  gravity 3 + previous action 22 + three pole XY offsets 6
* history              10 frames, oldest first
* goal                570 = 10 future reference frames x 57 physical values
* context             149 = randomization layout 143 + course poles 6
* action               22, scale 0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from supermimic.randomization import OBSERVATION_NOISE_STD
from supermimic.robot import Q1Robot, normalize_quat, quat_apply, quat_inverse, quat_mul

from .models import (
    SOURCE_ACTOR_HIDDEN,
    SOURCE_CRITIC_HIDDEN,
    SOURCE_MODEL_HIDDEN,
    build_source_models,
)

SOURCE_OBS_DIM = 78
SOURCE_HISTORY_FRAMES = 10
SOURCE_ACTION_DIM = 22
SOURCE_STATE_DIM = 79
SOURCE_PHYSICAL_DIM = 57
SOURCE_REFERENCE_DIM = 29
SOURCE_GOAL_DIM = SOURCE_PHYSICAL_DIM * SOURCE_HISTORY_FRAMES
SOURCE_CONTEXT_DIM = 149
SOURCE_POLE_FEATURES = 6
SOURCE_ACTION_SCALE = 0.5
SOURCE_CONTROL_DT = 0.02
SOURCE_PREVIOUS_ACTION = slice(50, 72)
SOURCE_POLE_SLICE = slice(72, 78)
SOURCE_DRIVE_CONTEXT_SCALE = 0.016


class PoleContextLayout(NamedTuple):
    """The source task's 149-column per-episode context layout.

    Column order is part of the checkpoint contract: the world model and critic
    consume the full vector, and the robot reads ``motor_offset``, ``delay_s``
    and ``course_poles`` by name.
    """

    friction: slice
    reserved_scalar: slice
    motor_offset: slice
    kp_factor: slice
    kd_factor: slice
    reserved_joints: slice
    root_com: slice
    torso_added_mass: slice
    reserved_bodies: slice
    delay_s: slice
    previous_executed_action: slice
    course_poles: slice
    size: int

    @staticmethod
    def build(
        joints: int,
        links: int,
        *,
        modeled_action_delay: bool = True,
        course_conditioned: bool = True,
    ) -> "PoleContextLayout":
        if joints != SOURCE_ACTION_DIM or links != 26:
            raise ValueError(
                f"Source pole context expects 22 joints and 26 links, got {joints} and {links}"
            )
        widths = (
            1,
            1,
            joints,
            joints,
            joints,
            joints,
            3,
            1,
            links,
            1 if modeled_action_delay else 0,
            joints if modeled_action_delay else 0,
            SOURCE_POLE_FEATURES if course_conditioned else 0,
        )
        bounds, start = [], 0
        for width in widths:
            bounds.append(slice(start, start + width))
            start += width
        layout = PoleContextLayout(*bounds, size=start)
        if layout.size != SOURCE_CONTEXT_DIM:
            raise ValueError(
                f"Source pole context must be {SOURCE_CONTEXT_DIM} columns, got {layout.size}"
            )
        return layout


class SourcePoleRobot(Q1Robot):
    """Q1 robot with the source task's 78-value observation and 149-value context."""

    def __init__(self, xml: Path, *, action_scale: float = SOURCE_ACTION_SCALE):
        super().__init__(
            xml,
            reward_scaling="q1_proportional",
            modeled_action_delay=True,
        )
        if self.model.nu != SOURCE_ACTION_DIM or self.model.nq != 29 or self.model.nv != 28:
            raise ValueError("Source pole robot requires the Q1 V3 22-actuator/29-qpos model")
        if not 0 < action_scale <= 1:
            raise ValueError("Source pole action scale must be in (0, 1]")
        # Replace the vendored 143-column layout with the source task's
        # course-conditioned 149-column layout.  Both share the leading slices.
        self.context_layout = PoleContextLayout.build(
            len(self.joint_names), len(self.body_names)
        )
        self.context_dim = self.context_layout.size
        self.obs_dim = SOURCE_OBS_DIM
        self.course_conditioned = True
        self.goal_representation = "robot"
        self.action_scale = float(action_scale)
        self.register_buffer("course_origin", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("course_yaw", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "course_pole_radius", torch.zeros((), dtype=torch.float32)
        )

    def set_course_frame(
        self,
        origin_xyz_m: tuple[float, float, float] | list[float],
        yaw_rad: float,
        pole_radius_m: float = 0.0,
    ) -> None:
        """Place the route frame in the world, as the source simulator does."""
        self.course_origin.copy_(self.course_origin.new_tensor(origin_xyz_m))
        self.course_yaw.copy_(self.course_yaw.new_tensor(yaw_rad))
        self.course_pole_radius.fill_(float(pole_radius_m))

    def observation(self, state: Tensor, context: Tensor) -> Tensor:
        """Return the source 78-value observation in the registered order."""
        if state.shape[-1] != SOURCE_STATE_DIM:
            raise ValueError(f"Source pole state must have {SOURCE_STATE_DIM} values")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"Source pole context must have {self.context_dim} values")
        q, dq = state[..., 7:29], state[..., 35:57]
        rotation = normalize_quat(state[..., 3:7])
        inverse = quat_inverse(rotation)
        gyro = quat_apply(inverse, state[..., 32:35])
        gravity = quat_apply(inverse, state.new_tensor([0.0, 0.0, -1.0]))
        features = (
            q - self.default_q - context[..., self.context_layout.motor_offset],
            dq,
            gyro,
            gravity,
            state[..., 57:79],
        )
        route_xy = context[..., self.context_layout.course_poles].reshape(
            *state.shape[:-1], 3, 2
        )
        cosine, sine = self.course_yaw.cos(), self.course_yaw.sin()
        world_x = (
            self.course_origin[0] + cosine * route_xy[..., 0] - sine * route_xy[..., 1]
        )
        world_y = (
            self.course_origin[1] + sine * route_xy[..., 0] + cosine * route_xy[..., 1]
        )
        world_z = self.course_origin[2].expand_as(world_x)
        poles_world = torch.stack((world_x, world_y, world_z), dim=-1)
        relative_world = poles_world - state[..., None, :3]
        relative_heading = quat_apply(inverse[..., None, :], relative_world)
        features += (relative_heading[..., :2].flatten(-2),)
        observation = torch.cat(features, dim=-1)
        if observation.shape[-1] != SOURCE_OBS_DIM:
            raise ValueError(
                f"Source pole observation must have {SOURCE_OBS_DIM} values, "
                f"got {observation.shape[-1]}"
            )
        return observation

    def previous_action_from_observation(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected {self.obs_dim} policy features, got {observation.shape[-1]}"
            )
        return observation[..., SOURCE_PREVIOUS_ACTION]

    def noise_scale(self) -> Tensor:
        """Observation noise: source sensor scales plus exact pole coordinates."""
        scale = self.default_q.new_tensor(OBSERVATION_NOISE_STD)
        if scale.numel() != SOURCE_OBS_DIM - SOURCE_POLE_FEATURES:
            raise ValueError("Vendored observation noise does not match the source layout")
        return torch.cat((scale, scale.new_zeros(SOURCE_POLE_FEATURES)))

    def pd_target(self, actions: Tensor, context: Tensor) -> Tensor:
        """Convert normalized policy actions to the source PD position target."""
        if actions.shape[-1] != SOURCE_ACTION_DIM:
            raise ValueError(f"Source pole actions must have {SOURCE_ACTION_DIM} values")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"Source pole context must have {self.context_dim} values")
        zero = self.default_q + context[..., self.context_layout.motor_offset]
        return zero + self.action_scale * actions

    def set_course_poles(self, context: Tensor, poles_xy: Tensor) -> Tensor:
        """Write ordered route-frame pole centers into the context columns."""
        if poles_xy.shape[-2:] != (3, 2):
            raise ValueError("Course poles must have shape (..., 3, 2)")
        context = context.clone()
        context[..., self.context_layout.course_poles] = poles_xy.flatten(-2)
        return context


def source_goal_features(robot: SourcePoleRobot, references: Tensor, state: Tensor) -> Tensor:
    """Encode ten future reference frames in the robot's current root frame.

    This is the source task's ``goal_representation = "robot"`` encoding.  It
    keeps the 570-value layout (10 x 57) of the registered warmup checkpoint
    while making the robot's own position and heading error observable.
    """
    if getattr(robot, "goal_representation", None) != "robot":
        raise ValueError("Source pole goals require robot-anchored reference features")
    if references.shape[-2] != SOURCE_HISTORY_FRAMES or references.shape[-1] < SOURCE_PHYSICAL_DIM:
        raise ValueError(
            "Source pole goals require ten paired reference frames of at least 57 "
            "physical state values"
        )
    if state.shape[-1] != SOURCE_STATE_DIM:
        raise ValueError(f"Source pole state must have {SOURCE_STATE_DIM} values")
    references = references[..., :SOURCE_PHYSICAL_DIM]
    inverse = quat_inverse(normalize_quat(state[..., 3:7])).unsqueeze(-2)
    rotations = quat_mul(inverse, references[..., 3:7])
    rotations = torch.where(rotations[..., :1] < 0, -rotations, rotations)
    goal = torch.cat(
        (
            quat_apply(inverse, references[..., :3] - state[..., None, :3]),
            rotations,
            references[..., 7:29] - robot.default_q,
            quat_apply(inverse, references[..., 29:32]) / 3,
            quat_apply(inverse, references[..., 32:35]) / 3,
            references[..., 35:57] / 10,
        ),
        dim=-1,
    ).flatten(-2)
    if goal.shape[-1] != SOURCE_GOAL_DIM:
        raise ValueError(
            f"Source pole goal must have {SOURCE_GOAL_DIM} values, got {goal.shape[-1]}"
        )
    return goal


def validate_source_history(history: Tensor) -> None:
    """Reject histories that do not match the source 10-frame contract."""
    if history.shape[-2:] != (SOURCE_HISTORY_FRAMES, SOURCE_OBS_DIM):
        raise ValueError(
            "Source pole history must be 10 frames by 78 observations in "
            "oldest-to-newest order"
        )


def predict_source_action(
    robot: SourcePoleRobot,
    actor: torch.nn.Module,
    history: Tensor,
    references: Tensor,
    state: Tensor,
) -> Tensor:
    """Run the source actor from history, a future reference window and state."""
    validate_source_history(history)
    return actor(history, source_goal_features(robot, references, state))


__all__ = [
    "PoleContextLayout",
    "SourcePoleRobot",
    "SOURCE_ACTION_DIM",
    "SOURCE_ACTION_SCALE",
    "SOURCE_ACTOR_HIDDEN",
    "SOURCE_CONTEXT_DIM",
    "SOURCE_CONTROL_DT",
    "SOURCE_CRITIC_HIDDEN",
    "SOURCE_GOAL_DIM",
    "SOURCE_HISTORY_FRAMES",
    "SOURCE_MODEL_HIDDEN",
    "SOURCE_OBS_DIM",
    "SOURCE_PHYSICAL_DIM",
    "SOURCE_POLE_FEATURES",
    "SOURCE_REFERENCE_DIM",
    "SOURCE_STATE_DIM",
    "build_source_models",
    "predict_source_action",
    "source_goal_features",
    "validate_source_history",
]
