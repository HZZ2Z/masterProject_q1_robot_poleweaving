"""Headless native MuJoCo course environment with no soccer-task logic."""

from __future__ import annotations

import numpy as np
import mujoco
import yaml

from . import SHARED_ROOT
from .data import CourseCase
from .scene import build_course_scene, verify_course_scene


class PoleEnvironment:
    control_dt = 0.02

    def __init__(self, case: CourseCase, scene_path, *, seed: int = 0):
        self.case = case
        self.rng = np.random.default_rng(seed)
        with (SHARED_ROOT / "supermimic/expert/configs/Q1.yaml").open() as stream:
            self.cfg = yaml.safe_load(stream)
        self.scene_path = build_course_scene(case, scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(self.cfg["sim"]["dt"])
        self.data = mujoco.MjData(self.model)
        joints = self.model.actuator_trnid[:, 0]
        names = [self.model.joint(int(index)).name for index in joints]
        if names != self.cfg["asset"]["dof_names"] or self.model.nu != 22:
            raise ValueError("Pole scene differs from the Q1 V3 22-action contract")
        verify_course_scene(self.model, case)
        roots = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.root_joint = int(roots[0])
        self.root_q = int(self.model.jnt_qposadr[self.root_joint])
        self.root_v = int(self.model.jnt_dofadr[self.root_joint])
        self.qpos_addrs = self.model.jnt_qposadr[joints]
        self.qvel_addrs = self.model.jnt_dofadr[joints]
        self.default = np.asarray(
            [self.cfg["init_state"]["default_joint_angles"].get(name, self.cfg["init_state"]["default_joint_angles"]["default"]) for name in names],
            dtype=np.float64,
        )
        self.action_scale = np.asarray(self.cfg["control"]["action_scale"], dtype=np.float64)
        self.kp = np.asarray([self._gain(name, "stiffness") for name in names])
        self.kd = np.asarray([self._gain(name, "damping") for name in names])
        self.decimation = int(self.cfg["control"]["decimation"])
        self.pole_ids = {int(self.model.geom(f"course_pole_{i}").id) for i in range(1, 4)}
        self.action = np.zeros(22, dtype=np.float32)
        self.history = np.zeros((50, 79), dtype=np.float32)
        self.frame = 0
        self.collision = False
        self.fall = False
        self.transitions = []
        self.root_xy = np.zeros(2)
        self.track_sq_error = []

    def _gain(self, joint: str, section: str) -> float:
        return float(next(value for key, value in self.cfg["control"][section].items() if key in joint))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.case.qpos[0]
        if len(self.case.qpos) > 1:
            mujoco.mj_differentiatePos(
                self.model,
                self.data.qvel,
                self.control_dt,
                self.case.qpos[0],
                self.case.qpos[1],
            )
        mujoco.mj_forward(self.model, self.data)
        self.action.fill(0)
        self.frame = 0
        self.collision = self.fall = False
        self.transitions.clear()
        self.track_sq_error.clear()
        obs = self.observation()
        self.history[:] = obs
        return obs.copy()

    def state(self) -> np.ndarray:
        qvel = self.data.qvel.copy()
        world_angular = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(world_angular, qvel[self.root_v + 3 : self.root_v + 6], self.data.qpos[self.root_q + 3 : self.root_q + 7])
        qvel[self.root_v + 3 : self.root_v + 6] = world_angular
        return np.concatenate((self.data.qpos.copy(), qvel, self.action.copy())).astype(np.float32)

    def observation(self) -> np.ndarray:
        q = self.data.qpos[self.root_q + 3 : self.root_q + 7]
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
        obs = np.zeros(79, dtype=np.float32)
        gravity = np.array([0.0, 0.0, -1.0])
        obs[:3] = _quat_rotate_inverse(q, gravity)
        obs[3:6] = self.data.qvel[self.root_v + 3 : self.root_v + 6]
        # The ball and soccer-goal slots remain zero; task state is carried by
        # explicit reference and pole features instead.
        obs[11:13] = [np.cos(yaw), np.sin(yaw)]
        obs[13:35] = self.data.qpos[self.qpos_addrs] - self.default
        obs[35:57] = self.data.qvel[self.qvel_addrs] * 0.1
        obs[57:79] = self.action
        return obs

    def step(self, action: np.ndarray):
        raw = np.asarray(action, dtype=np.float64).reshape(22)
        previous = self.state()
        obs = self.observation()
        clipped = np.clip(raw, -1.0, 1.0)
        target = np.clip(self.default + self.action_scale * clipped, self.model.jnt_range[self.model.actuator_trnid[:, 0], 0], self.model.jnt_range[self.model.actuator_trnid[:, 0], 1])
        self.action[:] = clipped
        self._step_pole_contacts = set()
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_addrs]) - self.kd * self.data.qvel[self.qvel_addrs]
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
            contacted_poles = set()
            for contact in self.data.contact:
                if contact.dist <= 0 and self.pole_ids.intersection((int(contact.geom1), int(contact.geom2))):
                    self.collision = True
                    contacted_poles.update(self.pole_ids.intersection((int(contact.geom1), int(contact.geom2))))
            if contacted_poles:
                self._step_pole_contacts.update(contacted_poles)
        self.frame = min(self.frame + 1, len(self.case.qpos) - 1)
        next_state = self.state()
        root = self.data.qpos[self.root_q : self.root_q + 3]
        target_pose = self.case.qpos[self.frame]
        root_error = float(np.square(root[:2] - target_pose[:2]).sum())
        self.track_sq_error.append(root_error)
        self.root_xy = root[:2].copy()
        w, x, y, z = self.data.qpos[self.root_q + 3 : self.root_q + 7]
        upright_z = 1 - 2 * (x * x + y * y)
        self.fall = bool(root[2] < 0.2 or upright_z < 0.5 or np.linalg.norm(self.data.qvel[self.root_v : self.root_v + 6]) > 50)
        tracking_error = np.concatenate((root[:2] - target_pose[:2], self.data.qpos[7:29] - target_pose[7:29]))
        tracking = np.exp(-float(np.mean(tracking_error ** 2)) / 0.25)
        clearance = float(np.linalg.norm(self.case.poles_xy - root[None, :2], axis=-1).min() - 0.25)
        clearance_penalty = min(clearance, 0.0) ** 2
        reward = float(0.02 * tracking - 0.1 * clearance_penalty - float(self.collision) - float(self.fall))
        cause = "pole_collision" if self.collision else "fall" if self.fall else "route_end" if self.frame == len(self.case.qpos) - 1 else None
        transition = {
            "state": previous,
            "observation": obs,
            "action": clipped.astype(np.float32),
            "next_state": next_state,
            "reference_id": self.case.identity,
            "pole_xy": self.case.poles_xy.copy(),
            "pole_collision": bool(self.collision),
            "pole_contacts": sorted(self.model.geom(geom_id).name for geom_id in self._step_pole_contacts),
            "terminal_cause": cause,
            "reward": reward,
        }
        self.transitions.append(transition)
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = self.observation()
        return self.history[0].copy(), reward, cause is not None, transition

    def result(self) -> dict:
        root = self.data.qpos[self.root_q : self.root_q + 3]
        endpoint_distance = float(np.linalg.norm(root[:2] - self.case.qpos[-1, :2]))
        rmse = float(np.sqrt(np.mean(self.track_sq_error))) if self.track_sq_error else float("inf")
        route_complete = self.frame == len(self.case.qpos) - 1 and endpoint_distance <= 0.5
        return {
            "case_id": self.case.case_id,
            "steps": len(self.transitions),
            "endpoint_distance_m": endpoint_distance,
            "route_completed": route_complete,
            "pole_collision": self.collision,
            "fall": self.fall,
            "root_xy_rmse_m": rmse,
            "success": route_complete and not self.collision and not self.fall and rmse <= 0.5,
        }


def _quat_rotate_inverse(q_wxyz, vector):
    w = q_wxyz[0]
    xyz = q_wxyz[1:]
    return vector * (2 * w * w - 1) - 2 * w * np.cross(xyz, vector) + 2 * xyz * np.dot(xyz, vector)
