"""MuJoCo Warp stepping only. No hidden resets or policy observations."""
from pathlib import Path

import mujoco
import numpy as np
import torch


SCENE = Path(__file__).resolve().parents[1] / "resources/q1_description/q1_soccer.xml"


class WarpPhysics:
    def __init__(self, count, device="cuda:0", scene=SCENE):
        import mujoco_warp as mjw
        import warp as wp

        if count < 1 or torch.device(device).type != "cuda":
            raise ValueError("WarpPhysics requires positive world count and a CUDA device")
        self.wp, self.mjw = wp, mjw
        wp.init()
        self.device = torch.device(device)
        self.native_model = mujoco.MjModel.from_xml_path(str(scene))
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.stream = wp.stream_from_torch(self.torch_stream)
        with wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.native_model)
            self.data = mjw.make_data(self.native_model, nworld=count, nconmax=128, njmax=512)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.graph = None

    @torch.no_grad()
    def reset(self, qpos, qvel):
        if qpos.shape != self.qpos.shape or qvel.shape != self.qvel.shape:
            raise ValueError("Reset state must match all worlds")
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), self.wp.ScopedStream(self.stream):
            self.qpos.copy_(qpos)
            self.qvel.copy_(qvel)
            self.ctrl.zero_()
            self.wp.to_torch(self.data.qacc_warmstart).zero_()
            self.wp.to_torch(self.data.time).zero_()
            self.mjw.forward(self.model, self.data)
        caller.wait_stream(self.torch_stream)

    @torch.no_grad()
    def reset_indices(self, indices, qpos, qvel):
        """Reset selected episodes; other worlds preserve positions and velocities."""
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), self.wp.ScopedStream(self.stream):
            self.qpos[indices] = qpos
            self.qvel[indices] = qvel
            self.ctrl[indices] = 0
            self.wp.to_torch(self.data.qacc_warmstart)[indices] = 0
            self.wp.to_torch(self.data.time)[indices] = 0
            self.mjw.forward(self.model, self.data)
        caller.wait_stream(self.torch_stream)

    @torch.no_grad()
    def ball_foot_contacts(self, ball_geom, foot_geoms):
        """Actual active penetrations, never a foot-distance proxy."""
        wp, data = self.wp, self.data
        geom = wp.to_torch(data.contact.geom)
        world = wp.to_torch(data.contact.worldid)
        distance = wp.to_torch(data.contact.dist)
        valid = torch.arange(geom.shape[0],device=self.device) < wp.to_torch(data.nacon)[0]
        ball = (geom == ball_geom).any(-1)
        foot = torch.zeros_like(ball)
        for identifier in foot_geoms:
            foot |= (geom == identifier).any(-1)
        active = (wp.to_torch(data.contact.efc_address) >= 0).any(-1)
        valid &= ball & foot & (distance <= 0) & active & (world >= 0) & (world < self.qpos.shape[0])
        result = torch.zeros(self.qpos.shape[0],device=self.device,dtype=torch.int32)
        result.scatter_add_(0,world.long().clamp(0,self.qpos.shape[0]-1),valid.int())
        return result > 0

    @torch.no_grad()
    def step(self, torque):
        if torque.shape != self.ctrl.shape:
            raise ValueError("Torque must match [world, actuator]")
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), self.wp.ScopedStream(self.stream):
            self.ctrl.copy_(torque)
            if self.graph is None:
                # Compile once outside capture. Capture records, but does not
                # launch: the warm-up is this call's single physics step.
                self.mjw.step(self.model, self.data)
                with self.wp.ScopedCapture(stream=self.stream) as capture:
                    self.mjw.step(self.model, self.data)
                self.graph = capture.graph
            else:
                self.wp.capture_launch(self.graph)
        caller.wait_stream(self.torch_stream)

    def check(self):
        if self.wp.to_torch(self.data.overflow).any().item():
            raise RuntimeError("MuJoCo Warp contact/constraint allocation overflow")
        if not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all():
            raise FloatingPointError("Non-finite simulator state")


def smoke():
    """Short native/Warp stepping check, NOT a training or standing test."""
    torch.cuda.set_per_process_memory_fraction(0.4)
    physics = WarpPhysics(4)
    model = physics.native_model
    qpos = torch.tensor(model.qpos0, dtype=torch.float32, device="cuda").repeat(4, 1)
    qpos[:, 2] += 1  # Free fall isolates stepping from contact-solver differences.
    qvel = torch.zeros((4, model.nv), device="cuda")
    physics.reset(qpos, qvel)
    native = mujoco.MjData(model)
    native.qpos[:] = qpos[0].cpu().numpy()
    mujoco.mj_forward(model, native)
    torque = torch.zeros((4, model.nu), device="cuda")
    for _ in range(20):
        physics.step(torque)
        mujoco.mj_step(model, native)
    physics.check()
    difference = np.max(np.abs(physics.qpos[0].cpu().numpy() - native.qpos))
    if difference > 2e-4:
        raise AssertionError(f"Native/Warp free-fall mismatch: {difference}")
    print(f"20 steps, 4 worlds: native/Warp max qpos difference={difference:.3g}")


if __name__ == "__main__":
    smoke()
