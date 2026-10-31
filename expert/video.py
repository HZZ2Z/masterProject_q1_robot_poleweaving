"""Stream one MP4 per playback episode without buffering frames in RAM."""
from datetime import datetime
from pathlib import Path

import mujoco


class EpisodeVideoRecorder:
    def __init__(self, sim, directory, draw_scene=None, fps=50, width=1280, height=720):
        import imageio.v2 as imageio

        self.sim = sim
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.draw_scene = draw_scene
        self.fps = fps
        self.imageio = imageio
        self.session = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.episode = 1
        self.writer = None
        self.path = None
        self.start_time = None
        self.last_time = None
        self.frames = 0
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 5
        self.camera.elevation = -20
        # Renderer dimensions must fit MuJoCo's offscreen framebuffer.
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(sim.model, height=height, width=width)
        print(f'Recording episodes to {self.directory.resolve()} ({width}x{height}, {fps} fps)', flush=True)

    def set_camera(self, camera):
        for field in ('type', 'fixedcamid', 'trackbodyid', 'distance', 'azimuth', 'elevation', 'orthographic'):
            setattr(self.camera, field, getattr(camera, field))
        self.camera.lookat[:] = camera.lookat

    def capture(self, force=False):
        now = float(self.sim.data.time)
        if self.last_time is not None and abs(now - self.last_time) < 1e-9:
            return
        if self.start_time is not None and not force and now - self.start_time + 1e-9 < self.frames / self.fps:
            return
        if self.writer is None:
            self.path = self.directory / f'{self.session}_episode_{self.episode:06d}.partial.mp4'
            self.writer = self.imageio.get_writer(
                str(self.path), fps=self.fps, codec='libx264', quality=8,
                pixelformat='yuv420p', macro_block_size=2, ffmpeg_log_level='error',
                output_params=['-movflags', '+faststart'])
            self.start_time = now
        self.camera.lookat[:] = self.sim.data.sensor('position').data
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        if self.draw_scene is not None:
            self.draw_scene(self.renderer.scene)
        self.writer.append_data(self.renderer.render())
        self.last_time = now
        self.frames += 1

    def finish_episode(self, cause):
        """Called on the terminal state BEFORE any robot/ball reset."""
        if self.writer is None:
            return
        try:
            self.capture(force=True)
        finally:
            self.writer.close()
            self.writer = None
        target = self.path.with_name(f'{self.session}_episode_{self.episode:06d}_{cause}.mp4')
        self.path.replace(target)
        print(f'Video saved: {target} ({self.frames} frames, end={cause})', flush=True)
        self.episode += 1
        self.start_time = self.last_time = None
        self.frames = 0

    def close(self, cause='stopped'):
        try:
            self.finish_episode(cause)
        finally:
            self.renderer.close()
