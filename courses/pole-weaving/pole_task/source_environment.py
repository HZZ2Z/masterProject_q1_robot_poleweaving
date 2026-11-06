"""Source task 7.7-compatible Q1 control on the soccer-field course scene.

The scene embeds the registered source robot XML, but the source task's
control loop is not part of that XML: it uses the walk/run PD table, a 0.5
action scale, the XML's ``qpos0`` zero pose and directional torque-speed
limits.  This module reproduces those semantics for headless CPU MuJoCo so
pole collection and evaluation cannot silently run the soccer expert's gains.
"""

from __future__ import annotations

import numpy as np
import mujoco
import torch

from supermimic.randomization import (
    RandomizationCfg,
    nominal_context,
    sample_context,
    sample_push,
)

from .data import CourseCase
from .scene import build_course_scene, verify_course_scene

SOURCE_CONTROL_DT = 0.02
SOURCE_PHYSICS_TIMESTEP = 0.002
SOURCE_ACTION_SCALE = 0.5

# Q1 walk/run PD calibration from the source simulator, matched as substrings
# of the XML joint name exactly as that simulator does.
SOURCE_STIFFNESS = {
    "hip_pitch_joint": 40.0,
    "hip_roll_joint": 40.0,
    "hip_yaw_joint": 40.0,
    "knee_joint": 40.0,
    "ankle_pitch_joint": 20.0,
    "ankle_roll_joint": 10.0,
    "waist_yaw_joint": 80.0,
    "waist_roll_joint": 80.0,
    "shoulder": 80.0,
    "elbow": 80.0,
}
SOURCE_DAMPING = {
    "hip_pitch_joint": 1.0,
    "hip_roll_joint": 1.0,
    "hip_yaw_joint": 1.0,
    "knee_joint": 1.0,
    "ankle_pitch_joint": 1.0,
    "ankle_roll_joint": 1.0,
    "waist_yaw_joint": 2.0,
    "waist_roll_joint": 2.0,
    "shoulder": 2.0,
    "elbow": 2.0,
}
SOURCE_BIG_MOTOR = (34.0, 7.5, 17.0)
SOURCE_SMALL_MOTOR = (21.0, 8.0, 16.0)


def _match(joint: str, table: dict[str, float], label: str) -> float:
    keys = [key for key in table if key in joint]
    if len(keys) != 1:
        raise ValueError(f"Joint {joint!r} matches {len(keys)} {label} keys, expected 1")
    return table[keys[0]]


def source_pd_gains(joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    kp = [_match(name, SOURCE_STIFFNESS, "PD stiffness") for name in joint_names]
    kd = [_match(name, SOURCE_DAMPING, "PD damping") for name in joint_names]
    return np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)


def source_torque_speed_parameters(
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peaks, knees, cutoffs = [], [], []
    for name in joint_names:
        big = any(part in name for part in ("hip_", "knee_", "waist_"))
        small = any(part in name for part in ("ankle_", "shoulder_", "elbow_"))
        if big == small:
            raise ValueError(f"Joint {name!r} has no unambiguous Q1 motor curve")
        peak, knee, cutoff = SOURCE_BIG_MOTOR if big else SOURCE_SMALL_MOTOR
        peaks.append(peak)
        knees.append(knee)
        cutoffs.append(cutoff)
    return (
        np.asarray(peaks, dtype=np.float64),
        np.asarray(knees, dtype=np.float64),
        np.asarray(cutoffs, dtype=np.float64),
    )


def source_torque_speed_limits(
    velocity: np.ndarray,
    peaks: np.ndarray,
    knees: np.ndarray,
    cutoffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Directional bounds: full braking torque remains at high speed."""
    slope = peaks / (cutoffs - knees)
    upper = np.clip(peaks - slope * (velocity - knees), 0.0, None)
    upper = np.minimum(upper, peaks)
    lower = np.clip(-peaks - slope * (velocity + knees), None, 0.0)
    lower = np.maximum(lower, -peaks)
    return lower, upper


class SourcePoleEnvironment:
    """Single-environment source Q1 control with paired course poles."""

    def __init__(
        self,
        case: CourseCase,
        scene_path,
        robot,
        *,
        seed: int = 0,
        randomize: bool = False,
    ):
        self.case = case
        self.robot = robot
        self.rng = np.random.default_rng(seed)
        self.randomize = bool(randomize)
        self.randomization = RandomizationCfg()
        self.scene_path = build_course_scene(case, scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = SOURCE_PHYSICS_TIMESTEP
        substeps = SOURCE_CONTROL_DT / SOURCE_PHYSICS_TIMESTEP
        if not np.isclose(substeps, round(substeps)):
            raise ValueError("Source physics timestep must divide the 50 Hz control period")
        self.decimation = int(round(substeps))
        self.data = mujoco.MjData(self.model)

        joints = self.model.actuator_trnid[:, 0]
        names = tuple(self.model.joint(int(index)).name for index in joints)
        if names != tuple(robot.joint_names) or self.model.nu != 22:
            raise ValueError("Course scene actuators differ from the source Q1 joint order")
        verify_course_scene(self.model, case)

        roots = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(roots) != 1:
            raise ValueError("Source course scene must have exactly one free root joint")
        self.root_joint = int(roots[0])
        self.root_q = int(self.model.jnt_qposadr[self.root_joint])
        self.root_v = int(self.model.jnt_dofadr[self.root_joint])
        self.qpos_addrs = self.model.jnt_qposadr[joints]
        self.qvel_addrs = self.model.jnt_dofadr[joints]
        self.default = robot.default_q.detach().cpu().numpy().astype(np.float64)
        if self.default.shape != (22,):
            raise ValueError("Source robot default pose must have 22 joints")
        self.action_scale = SOURCE_ACTION_SCALE
        self.kp, self.kd = source_pd_gains(names)
        self.torque_peaks, self.torque_knees, self.torque_cutoffs = (
            source_torque_speed_parameters(names)
        )
        self.actuator_gear = self.model.actuator_gear[:, 0].copy()
        self.ctrl_limits = self.model.actuator_ctrlrange.copy()
        self.pole_ids = {
            int(self.model.geom(f"course_pole_{index}").id) for index in range(1, 4)
        }
        self.root_body = int(robot.root_id)
        self.torso_body = int(robot.torso_id)
        self.default_mass = self.model.body_mass.copy()
        self.default_ipos = self.model.body_ipos.copy()
        self.default_friction = self.model.geom_friction.copy()
        self.foot_geoms = np.asarray(
            [
                index
                for index, body in enumerate(self.model.geom_bodyid)
                if self.model.body(int(body)).name
                in ("left_ankle_roll_link", "right_ankle_roll_link")
            ],
            dtype=np.int64,
        )
        self.floor_geoms = np.asarray(
            [
                index
                for index, body in enumerate(self.model.geom_bodyid)
                if body == 0
                and self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE
            ],
            dtype=np.int64,
        )
        if len(self.foot_geoms) == 0 or len(self.floor_geoms) == 0:
            raise ValueError("Source randomization needs foot and floor collision geoms")
        self.layout = robot.context_layout
        self.context = torch.from_numpy(
            nominal_context(self.layout, 1, "cpu").numpy().copy()
        )
        self.zero = self.default.copy()
        self.kp_effective = self.kp.copy()
        self.kd_effective = self.kd.copy()
        self.delay_steps = 0
        self.last_action = np.zeros(22, dtype=np.float64)
        self.push_enabled = False
        self.push_timer = np.inf
        self.pushed = False
        self.frame = 0
        self.action = np.zeros(22, dtype=np.float64)
        self.collision = False
        self.pole_contacts: set[int] = set()
        self.steps = 0

    def _sample_context(self) -> torch.Tensor:
        """Draw one randomized context without disturbing the global torch RNG."""
        seed = int(self.rng.integers(0, 2**63 - 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return sample_context(self.randomization, self.layout, 1, "cpu")

    def _sample_push(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**63 - 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return sample_push(self.randomization, 1, "cpu")[0].numpy()

    def _apply_context_physics(self) -> None:
        """Apply startup mass, COM, and friction, then rebuild constants."""
        context = self.context[0].numpy()
        mass = self.default_mass.copy()
        mass[self.torso_body] += context[self.layout.torso_added_mass][0]
        self.model.body_mass[:] = mass
        self.model.body_ipos[self.root_body] = (
            self.default_ipos[self.root_body] + context[self.layout.root_com]
        )
        friction = self.default_friction.copy()
        if self.randomize:
            friction[self.foot_geoms, 0] = context[self.layout.friction][0]
            # Contact friction takes max(foot, floor); cap the floor so the
            # requested minimum still bites.
            friction[self.floor_geoms, 0] = np.minimum(
                friction[self.floor_geoms, 0], self.randomization.friction[0]
            )
        self.model.geom_friction[:] = friction
        mujoco.mj_setConst(self.model, self.data)

    @property
    def last_frame(self) -> int:
        return len(self.case.qpos) - 1

    def reset(self, frame: int = 0):
        if not 0 <= frame <= self.last_frame:
            raise ValueError(f"Reset frame {frame} is outside {self.case.case_id}")
        self.model.body_mass[:] = self.default_mass
        self.model.body_ipos[:] = self.default_ipos
        self.model.geom_friction[:] = self.default_friction
        mujoco.mj_resetData(self.model, self.data)
        poles = torch.as_tensor(
            self.case.poles_xy, dtype=torch.float32
        ).unsqueeze(0)
        context = (
            self._sample_context()
            if self.randomize
            else nominal_context(self.layout, 1, "cpu")
        )
        self.context = self.robot.set_course_poles(context, poles)
        self._apply_context_physics()
        # Source episode startup: deterministic reset pose and reference
        # velocity, then the calibrated zero and randomized PD factors.
        self.zero = self.default + self.context[0, self.layout.motor_offset].numpy()
        self.kp_effective = self.kp * self.context[0, self.layout.kp_factor].numpy()
        self.kd_effective = self.kd * self.context[0, self.layout.kd_factor].numpy()
        self.delay_steps = int(
            round(float(self.context[0, self.layout.delay_s]) / SOURCE_PHYSICS_TIMESTEP)
        )
        self.push_enabled = self.randomize
        self.push_timer = float(
            self.rng.uniform(*self.randomization.push_interval_seconds)
        )
        self.pushed = False
        self.last_action.fill(0.0)
        self.data.qpos[:] = self.case.qpos[frame]
        # Q1Motions stores reference velocities from a central difference, so
        # reset from the same convention rather than a one-sided difference.
        lo, hi = max(0, frame - 1), min(self.last_frame, frame + 1)
        if hi > lo:
            mujoco.mj_differentiatePos(
                self.robot.model,
                self.data.qvel,
                (hi - lo) * self.robot_control_dt,
                self.case.qpos[lo],
                self.case.qpos[hi],
            )
        mujoco.mj_forward(self.model, self.data)
        self.action.fill(0.0)
        self.frame = frame
        self.steps = 0
        self.collision = False
        self.pole_contacts.clear()
        return self.state()

    @property
    def robot_control_dt(self) -> float:
        return SOURCE_CONTROL_DT

    def state(self) -> np.ndarray:
        qvel = self.data.qvel.copy()
        world_angular = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            world_angular,
            qvel[self.root_v + 3 : self.root_v + 6],
            self.data.qpos[self.root_q + 3 : self.root_q + 7],
        )
        qvel[self.root_v + 3 : self.root_v + 6] = world_angular
        return np.concatenate(
            (self.data.qpos.copy(), qvel, self.action.copy())
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        """Apply one source PD control step; return the next state and contacts."""
        raw = np.asarray(action, dtype=np.float64).reshape(22)
        if not np.isfinite(raw).all():
            raise FloatingPointError("Nonfinite source pole action")
        previous = self.state()
        self.action[:] = raw
        target = self.zero + self.action_scale * raw
        previous_target = self.zero + self.action_scale * self.last_action
        contacts: set[int] = set()
        for substep in range(self.decimation):
            applied = target if substep >= self.delay_steps else previous_target
            velocity = self.data.qvel[self.qvel_addrs]
            torque = self.kp_effective * (applied - self.data.qpos[self.qpos_addrs])
            torque -= self.kd_effective * velocity
            lower, upper = source_torque_speed_limits(
                velocity, self.torque_peaks, self.torque_knees, self.torque_cutoffs
            )
            torque = np.clip(torque, lower, upper)
            self.data.ctrl[:] = np.clip(
                torque / self.actuator_gear,
                self.ctrl_limits[:, 0],
                self.ctrl_limits[:, 1],
            )
            mujoco.mj_step(self.model, self.data)
            for contact in self.data.contact:
                if contact.dist > 0:
                    continue
                hit = self.pole_ids.intersection((int(contact.geom1), int(contact.geom2)))
                if hit:
                    self.collision = True
                    contacts.update(hit)
        self.pole_contacts.update(contacts)
        self.last_action[:] = raw
        self.frame = min(self.frame + 1, self.last_frame)
        self.steps += 1
        transition_state = self.state()
        self._push_velocity()
        next_state = self.state()
        if not np.isfinite(next_state).all():
            raise FloatingPointError("Nonfinite source pole simulator state")
        transition = {
            "state": previous,
            "next_state": transition_state,
            "action": raw.astype(np.float32),
            "reference_id": self.case.identity,
            "pole_xy": self.case.poles_xy.copy(),
            "pole_collision": self.collision,
            "pole_contacts": sorted(
                self.model.geom(int(geom_id)).name for geom_id in contacts
            ),
            "pushed": self.pushed,
        }
        return next_state, transition

    def _push_velocity(self) -> None:
        """Apply the source's exogenous startup-time velocity impulse."""
        self.pushed = False
        if not self.push_enabled:
            return
        self.push_timer -= SOURCE_CONTROL_DT
        if self.push_timer > 0:
            return
        impulse = self._sample_push()
        self.data.qvel[self.root_v : self.root_v + 3] += impulse[:3]
        angular = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            angular,
            impulse[3:6],
            self.data.qpos[self.root_q + 3 : self.root_q + 7],
        )
        self.data.qvel[self.root_v + 3 : self.root_v + 6] += angular
        self.push_timer = float(
            self.rng.uniform(*self.randomization.push_interval_seconds)
        )
        self.pushed = True


__all__ = [
    "SOURCE_ACTION_SCALE",
    "SOURCE_CONTROL_DT",
    "SOURCE_PHYSICS_TIMESTEP",
    "SourcePoleEnvironment",
    "source_pd_gains",
    "source_torque_speed_limits",
    "source_torque_speed_parameters",
]
