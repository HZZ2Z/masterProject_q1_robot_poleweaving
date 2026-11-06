"""Q1 V3 scene, bounded PD contract and differentiable expert camera interface."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import torch
import yaml

from supermimic.robot import Q1Robot, normalize_quat, quat_apply, quat_inverse, quat_mul
from expert.runtime import ROOT as EXPERT_ROOT


class SoccerRobot(Q1Robot):
    """Vendor FK on the V3 skeleton; physical state includes the free ball.

    State: robot qpos29, world root velocity6, joint velocity22, previous
    action22, ball qpos7, world ball velocity6, native cached sensors17.
    Context: camera queue60,
    integer observation delay, camera phase. Actor never receives the state.
    """

    def __init__(self):
        scene = EXPERT_ROOT / "resources/q1_description/q1_soccer.xml"
        document = ET.parse(scene)
        tree = document.getroot()
        # Vendor Q1Robot validates a robot-only free skeleton. This temporary
        # view changes no robot parameters; native collection uses the full scene.
        world = tree.find("worldbody")
        world.remove(world.find("body[@name='ball']"))
        tree.find("compiler").set("meshdir", str(scene.parent / "meshes"))
        for texture in tree.findall(".//texture[@file]"):
            texture.set("file", str(scene.parent / texture.get("file")))
        with tempfile.TemporaryDirectory(prefix="q1-skeleton-") as folder:
            xml = Path(folder) / "robot.xml"
            document.write(xml)
            super().__init__(xml)
        self.cfg = yaml.safe_load((EXPERT_ROOT / "configs/Q1.yaml").read_text())
        names = self.joint_names
        cfg = self.cfg
        defaults = cfg["init_state"]["default_joint_angles"]
        neutral = [defaults.get(name, defaults["default"]) for name in names]
        self.register_buffer("neutral", torch.tensor(neutral, dtype=torch.float32))
        self.register_buffer(
            "action_scale", torch.tensor(cfg["control"]["action_scale"])
        )
        self.state_dim, self.obs_dim, self.context_dim = 109, 79, 62
        self.camera_site = self.model.site("policy_camera").id
        self.camera_body = int(self.model.site_bodyid[self.camera_site])
        self.register_buffer(
            "camera_position",
            torch.tensor(self.model.site_pos[self.camera_site], dtype=torch.float32),
        )
        self.register_buffer(
            "camera_rotation",
            torch.tensor(self.model.site_quat[self.camera_site], dtype=torch.float32),
        )
        self.register_buffer(
            "half_fov", torch.deg2rad(torch.tensor(cfg["asset"]["camera_fov"])) / 2
        )

    def bounded_action(self, raw):
        action = raw.clamp(
            -self.cfg["normalization"]["clip_actions"],
            self.cfg["normalization"]["clip_actions"],
        )
        target = (self.neutral + self.action_scale * action).clamp(
            self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        return action, target

    def observe(self, state, context, *, randomize=True, innovation=None):
        """Advance one causal sensor frame, including 25 Hz refresh and delay.

        The optional fixed innovation (visibility uniform and two normal draws)
        allows numeric comparison with native MuJoCo without changing its RNG.
        """
        # MuJoCo mj_step leaves its sensor cache at the beginning of the last
        # physics substep. These 17 learned state features preserve that phase;
        # recomputing FK from end-of-step qpos would advance sensors by 2 ms.
        rotation = normalize_quat(state[..., 95:99])
        w, x, y, z = rotation.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
        c, s = yaw.cos(), yaw.sin()

        def yaw_inverse(vector):
            return torch.stack(
                (
                    c * vector[..., 0] + s * vector[..., 1],
                    -s * vector[..., 0] + c * vector[..., 1],
                ),
                -1,
            )

        ball = yaw_inverse(state[..., 79:81] - state[..., 92:94])
        goal = yaw_inverse(state.new_tensor([7.0, 0.0]) - state[..., 92:94])
        camera_q = normalize_quat(state[..., 105:109])
        camera_p = state[..., 102:105]
        optical_q = quat_mul(camera_q, state.new_full((len(state), 4), 0.5))
        optical_ball = quat_apply(quat_inverse(optical_q), state[..., 79:82] - camera_p)
        cone = (
            (optical_ball[..., 2] > 0)
            & (
                torch.atan2(optical_ball[..., 0], optical_ball[..., 2]).abs()
                < self.half_fov[0]
            )
            & (
                torch.atan2(optical_ball[..., 1], optical_ball[..., 2]).abs()
                < self.half_fov[1]
            )
        )
        distance = ball.norm(dim=-1)
        if innovation is None:
            innovation = torch.cat(
                (torch.rand_like(distance)[:, None], torch.randn_like(ball)), -1
            )
        probability = (2.475 - 0.225 * distance.clamp_min(7.0)).clamp_min(0.0)
        visible = (
            torch.where(
                cone,
                innovation[:, 0] < probability,
                (ball[:, 0] > -2) & (innovation[:, 0] < 0.01),
            )
            if randomize
            else cone
        )
        noisy_ball = (
            ball + innovation[:, 1:] * (0.149 + 0.124 * distance[:, None])
            if randomize
            else ball
        )
        measured = torch.cat(
            (
                torch.where(visible[:, None], noisy_ball, 0.0),
                visible[:, None].to(state.dtype),
            ),
            -1,
        )
        queue = context[:, :60].reshape(-1, 20, 3)
        # Context stores the queue after the current observation. Next phase is
        # current phase + 1, hence refresh when current phase == 1.
        measured = torch.where((context[:, 61] == 1)[:, None], measured, queue[:, 0])
        queue = torch.cat((measured[:, None], queue[:, :-1]), 1)
        delay = context[:, 60].long().clamp(0, 19)
        delayed = queue[torch.arange(len(state), device=state.device), delay]
        norm = self.cfg["normalization"]
        obs = torch.cat(
            (
                quat_apply(quat_inverse(rotation), state.new_tensor([0.0, 0.0, -1.0]))
                * norm["gravity"],
                state[:, 99:102] * norm["ang_vel"],
                delayed,
                goal,
                torch.stack((c, s), -1),
                (state[:, 7:29] - self.neutral) * norm["dof_pos"],
                state[:, 35:57] * norm["dof_vel"],
                state[:, 57:79],
            ),
            -1,
        )
        next_context = torch.cat(
            (queue.flatten(1), context[:, 60:61], 1 - context[:, 61:62]), -1
        )
        return obs, next_context

    def reward(self, state, next_state, target, action, dt, target_heads=None):
        # Success motions are privileged training references only. Ball progress
        # and approach also provide a task objective when the rollout deviates.
        track = (
            -(next_state[:, 7:29] - target[:, 7:29]).square().mean(-1) / 0.25
        ).exp()
        gravity = quat_apply(
            quat_inverse(normalize_quat(next_state[:, 3:7])),
            state.new_tensor([0.0, 0.0, -1.0]),
        )
        upright = (-gravity[:, 2]).clamp(0, 1)
        goal = state.new_tensor([7.0, 0.0])
        progress = (state[:, 79:81] - goal).norm(dim=-1) - (
            next_state[:, 79:81] - goal
        ).norm(dim=-1)
        approach = (state[:, :2] - state[:, 79:81]).norm(dim=-1) - (
            next_state[:, :2] - next_state[:, 79:81]
        ).norm(dim=-1)
        smooth = (action - state[:, 57:79]).square().mean(-1)
        return dt * (track + upright - 0.05 * smooth) + 3 * progress + 0.5 * approach


def physical_states(arrays, sensors):
    """Convert MuJoCo local free-joint angular velocities to vendor world frame."""
    q = torch.tensor(arrays["qpos"], dtype=torch.float32)
    v = torch.tensor(arrays["qvel"], dtype=torch.float32)
    previous = torch.cat(
        (
            torch.tensor(arrays["observations"][:, 57:79]),
            torch.tensor(arrays["actions"][-1:]),
        ),
        0,
    )
    return torch.cat(
        (
            q[:, :29],
            v[:, :3],
            quat_apply(q[:, 3:7], v[:, 3:6]),
            v[:, 6:28],
            previous,
            q[:, 29:36],
            v[:, 28:31],
            quat_apply(q[:, 32:36], v[:, 31:34]),
            torch.tensor(sensors, dtype=torch.float32),
        ),
        -1,
    )
