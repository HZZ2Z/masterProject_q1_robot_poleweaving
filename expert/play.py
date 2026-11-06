"""Standalone expert playback. No parent workspace imports."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import mujoco.viewer
from .runtime import DEFAULT_CHECKPOINT,ROOT,SoccerPlayer,draw_camera_frustum


@contextmanager
def managed_viewer(model, data, **kwargs):
    """Wait for this passive viewer's render thread before Python/GLFW teardown.

    MuJoCo 3.17 Handle.close() requests exit asynchronously. Its daemon render
    thread must finish before glfw.terminate runs during interpreter shutdown.
    Other viewer threads are deliberately left alone.
    """
    existing = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, **kwargs)
    render_threads = [thread for thread in threading.enumerate()
                      if thread not in existing and thread.name.endswith('(_launch_internal)')]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in render_threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer render thread did not stop')


def run(checkpoint=None, headless=False, steps=None, seed=42, reset_mode='reference', task='Q1',
        show_frustum=True, frustum_range=2.5, out_video=None, sim=None):
    sim = SoccerPlayer(checkpoint, seed, reset_mode, task) if sim is None else sim
    limit = steps if steps is not None else (5000 if headless else None)
    visible = [show_frustum]
    recorder = None
    if out_video is not None:
        from .video import EpisodeVideoRecorder
        def decorate(scene):
            if visible[0]:
                draw_camera_frustum(scene, sim.model, sim.data, sim.cfg, frustum_range)
        recorder = EpisodeVideoRecorder(sim, out_video, draw_scene=decorate)
    on_episode_end = recorder.finish_episode if recorder is not None else None
    end_cause = 'interrupted'
    try:
        if headless:
            if recorder is not None:
                recorder.capture()
            for _ in range(limit):
                sim.step(on_episode_end=on_episode_end)
                if recorder is not None:
                    recorder.capture()
        else:
            def key_callback(key):
                if key in (ord('F'), ord('f')):
                    visible[0] = not visible[0]
            with managed_viewer(sim.model, sim.data, key_callback=key_callback) as viewer:
                viewer.cam.distance = 5
                viewer.cam.elevation = -20
                if recorder is not None:
                    with viewer.lock():
                        recorder.set_camera(viewer.cam)
                    recorder.capture()
                count = 0
                while viewer.is_running() and (limit is None or count < limit):
                    start = time.time()
                    if recorder is not None:
                        with viewer.lock():
                            recorder.set_camera(viewer.cam)
                    sim.step(on_episode_end=on_episode_end)
                    viewer.cam.lookat[:] = sim.data.sensor('position').data
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        if visible[0]:
                            draw_camera_frustum(viewer.user_scn, sim.model, sim.data, sim.cfg, frustum_range)
                    viewer.sync()
                    if recorder is not None:
                        recorder.capture()
                    count += 1
                    remaining = sim.model.opt.timestep - (time.time() - start)
                    if remaining > 0:
                        time.sleep(remaining)
        end_cause = 'stopped'
    except KeyboardInterrupt:
        print('Playback interrupted; saving the current episode.', flush=True)
    finally:
        if recorder is not None:
            recorder.close(end_cause)
    print(f'Steps: {sim.total_steps}, Goals: {sim.goals}, Out: {sim.outs}, Falls: {sim.falls}')
    return sim


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['Q1'],default='Q1')
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reset-mode',choices=['reference','default'],default='reference')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--out-video',type=Path)
    parser.add_argument('--no-frustum',action='store_true')
    parser.add_argument('--frustum-range',type=float,default=2.5)
    args=parser.parse_args()
    if args.frustum_range<=0:parser.error('frustum-range must be positive')
    run(args.checkpoint,args.headless,args.steps,args.seed,args.reset_mode,args.task,
        not args.no_frustum,args.frustum_range,args.out_video)


if __name__=='__main__':main()
