import argparse
import os
import sys
import json
# GLFW owns the interactive window; EGL is only used for offscreen rendering.
_offscreen = any(x.split('=')[0] in ('--headless','--image','--video') for x in sys.argv[1:])
os.environ.setdefault('MUJOCO_GL', 'egl' if _offscreen else 'glfw')
import time
import threading
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v3 as iio
from q1_4v4.env import SoccerEnv
from q1_4v4.learning import make_args, make_team, load_team, numpy
from q1_4v4.curriculum import STAGES
from q1_4v4.course_config import COURSES


def course_argument(value):
    if value.isdigit():
        index=int(value)
        if 0 <= index < len(COURSES): return COURSES[index]
    if value in COURSES: return value
    if value == 'kick': return 'kick_left'
    raise argparse.ArgumentTypeError('课程须为0–27或课程名称，使用 --list-courses 查看')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', help='Weights; defaults to current active training last.pt')
    p.add_argument('--course', '--stage', dest='course', type=course_argument, help='Course name or index 0–27')
    p.add_argument('--list-courses', action='store_true', help='List supported course names and exit')
    p.add_argument('--neutral', action='store_true', help='Preview neutral PD without loading a policy')
    p.add_argument('--device', default='cuda:0', help='Warp preview device')
    p.add_argument('--steps', type=int, default=None, help='Control steps; GUI defaults to unlimited, headless to 300')
    p.add_argument('--speed', type=float, default=1., help='GUI playback speed, e.g. 0.5')
    p.add_argument('--image', help='Save initial scene PNG and exit')
    p.add_argument('--video', help='Save headless MP4')
    p.add_argument('--headless', action='store_true')
    a = p.parse_args()
    if a.list_courses:
        for i,name in enumerate(COURSES): print(f'{i:2d}  {name}')
        return
    if a.steps is not None and a.steps < 0: p.error('--steps must be >=0')
    if a.speed <= 0: p.error('--speed must be positive')
    if a.neutral and a.checkpoint: p.error('--neutral and --checkpoint cannot be combined')
    if a.headless and a.steps == 0: p.error('--headless requires a finite number of steps')
    a.steps = a.steps if a.steps is not None else (300 if a.headless or a.video else 0)
    if not a.neutral and not a.checkpoint:
        record=Path(__file__).resolve().parent/'logs/active_training.json'
        if record.exists():
            candidate=Path(json.loads(record.read_text())['output'])/'last.pt'
            if candidate.exists(): a.checkpoint=str(candidate)
    torch.set_num_threads(2)
    ckpt = torch.load(a.checkpoint, weights_only=False, map_location='cpu') if a.checkpoint else None
    v2=ckpt is not None and ckpt.get('reward_version',1)>=2
    stage = a.course or (ckpt['curriculum']['stage'] if v2 else STAGES[ckpt['curriculum']['index']] if ckpt else 'match')
    if v2 or a.course is not None or ckpt is None:
        from q1_4v4.preview import CoursePreview
        env = CoursePreview(stage,device=a.device)
    else:
        env = SoccerEnv(stage=stage)
    print({'course':stage,'checkpoint':a.checkpoint,'control':'policy' if ckpt else 'neutral PD',
           'steps':a.steps or 'until GUI closes'},flush=True)
    teams = []
    if ckpt:
        args = make_args()
        args.hidden_size = ckpt['args']['hidden_size']
        for state in ckpt['teams']:
            policy, trainer, _ = make_team(args, env, torch.device('cpu'), gpu_rollout=ckpt.get('backend') == 'warp')
            load_team(state, policy, trainer)
            trainer.prep_rollout()
            teams.append(policy)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .2]
    camera.distance = 18
    camera.azimuth = 90
    camera.elevation = -65
    visual = mujoco.MjvOption()
    visual.geomgroup[3:5] = 0
    renderer = None
    viewer = None
    viewer_threads = []
    video = None
    try:
        if a.image or a.video:
            renderer = mujoco.Renderer(env.model, height=720, width=1280)
            renderer.update_scene(env.data, camera=camera, scene_option=visual)
            if a.image:
                Path(a.image).parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(a.image, renderer.render())
                return
            import imageio.v2 as imageio
            Path(a.video).parent.mkdir(parents=True, exist_ok=True)
            video = imageio.get_writer(a.video, fps=25)
        elif not a.headless:
            import mujoco.viewer as mjviewer
            prior_threads = set(threading.enumerate())
            viewer = mjviewer.launch_passive(env.model, env.data)
            viewer_threads = [t for t in threading.enumerate() if t not in prior_threads and '_launch_internal' in t.name]
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[:] = visual.geomgroup
        step = 0
        while a.steps == 0 or step < a.steps:
            tic = time.perf_counter()
            actions = np.zeros((8, 22), np.float32)
            if teams:
                with torch.no_grad():
                    for t, policy in enumerate(teams):
                        actions[t*4:t*4+4] = numpy(policy.act(env.observe()[t*4:t*4+4],
                            np.zeros((4, 1, args.hidden_size), np.float32), np.ones((4, 1), np.float32),
                            deterministic=True)[0])
            _, _, _, done, truncated, info = env.step(actions)
            if video and step % 2 == 0:
                renderer.update_scene(env.data, camera=camera, scene_option=visual)
                video.append_data(renderer.render())
            if viewer:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, env.dt/a.speed-(time.perf_counter()-tic)))
            if done or truncated:
                print(info, flush=True)
                env.reset()
            step += 1
        print({'steps': step, 'stage': stage, 'checkpoint': a.checkpoint})
    finally:
        if video:
            video.close()
        if viewer:
            viewer.close()
            # MuJoCo starts a daemon GUI thread; let it release its GL context
            # before Python tears down GLFW and GPU resources.
            for thread in viewer_threads:
                thread.join(timeout=5.)
        if renderer:
            renderer.close()


if __name__ == '__main__':
    main()
