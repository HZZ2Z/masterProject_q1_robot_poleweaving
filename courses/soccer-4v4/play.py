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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import mujoco
import imageio.v3 as iio
from q1_4v4.env import SoccerEnv
from q1_4v4.learning import make_args, make_team, load_team, numpy, with_team_identity
from q1_4v4.curriculum import STAGES
from q1_4v4.course_config import COURSES


def apply_checkpoint_motion_constraints(env, checkpoint):
    match_task = getattr(getattr(env, 'env', None), 'match_task', None)
    if match_task is None:
        return False
    motion = match_task.motion
    cfg = checkpoint.get('motion_constraints', {})
    motion.enabled = cfg.get('enabled', False)
    motion.start_iteration = cfg.get('start_iteration', 0)
    motion.iteration = checkpoint['iteration']
    return True


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
    p.add_argument('--match-layout',choices=('curriculum','standard'),default='curriculum',
                   help='Preview current near-ball kickoff curriculum or standard full-pitch kickoff')
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
        record=PROJECT_ROOT/'logs/active_training.json'
        if record.exists():
            candidate=Path(json.loads(record.read_text())['output'])/'last.pt'
            if candidate.exists(): a.checkpoint=str(candidate)
    torch.set_num_threads(2)
    ckpt = torch.load(a.checkpoint, weights_only=False, map_location='cpu') if a.checkpoint else None
    v2=ckpt is not None and ckpt.get('reward_version',1)>=2
    stage = a.course or (ckpt['curriculum']['stage'] if v2 else STAGES[ckpt['curriculum']['index']] if ckpt else 'match')
    if v2 or a.course is not None or ckpt is None:
        from q1_4v4.preview import CoursePreview
        from q1_4v4.course_config import marching_observation_mode
        env = CoursePreview(stage,device=a.device,marching_target_observation=marching_observation_mode(checkpoint=ckpt),
            match_control=ckpt.get('match_controls',{}).get(stage) if ckpt else None,match_layout=a.match_layout,
            match_gait_reward=ckpt.get('match_gait_reward',True) if ckpt else True)
        if ckpt:
            apply_checkpoint_motion_constraints(env, ckpt)
    else:
        env = SoccerEnv(stage=stage)
    print({'course':stage,'checkpoint':a.checkpoint,'control':'policy' if ckpt else 'neutral PD',
           'steps':a.steps or 'until GUI closes'},flush=True)
    shared = ckpt is not None and ckpt.get('policy_sharing')=='shared'
    teams = []
    if ckpt:
        args = make_args()
        args.hidden_size = ckpt['args']['hidden_size']
        args.marching_exploration_revision=ckpt.get('marching_exploration_revision',0)
        args.marching_exploration_mix=ckpt.get('marching_exploration_mix',.25)
        args.match_exploration_revision=ckpt.get('match_exploration_revision',0)
        args.history_frames=ckpt['args'].get('history_frames',0)
        args.history_bins=ckpt['args'].get('history_bins',5)
        args.amp_groups=ckpt['args'].get('amp_groups',False)
        for state in ckpt['teams']:
            policy, trainer, _ = make_team(args, env, torch.device('cpu'), gpu_rollout=ckpt.get('backend') == 'warp',shared=shared)
            load_team(state, policy, trainer,match_features=stage in ('teamplay','match'))
            trainer.prep_rollout()
            teams.append(policy)
    history=None
    if teams and args.history_frames:
        from q1_4v4.history import ActorHistory,history_payload
        history=ActorHistory(1,8,env.observation_space.shape[0],args.history_frames,args.history_bins,'cpu')
        history.push(torch.as_tensor(env.observe(),dtype=torch.float32)[None])
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
                        sl=slice(None) if shared else slice(t*4,t*4+4)
                        count=8 if shared else 4
                        obs=env.observe()[sl]
                        if shared:obs=with_team_identity(obs,np.arange(8)//4)
                        if history is not None:
                            obs=history_payload(torch.as_tensor(obs),history.pooled()[0,sl],shared,
                                                torch.arange(8)[sl]//4)
                        actions[sl] = numpy(policy.act(obs,
                            np.zeros((count, 1, args.hidden_size), np.float32), np.ones((count, 1), np.float32),
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
            elif info.get('reset_agents'):
                print(info, flush=True)
            if history is not None:
                reset=torch.zeros((1,8),dtype=torch.bool)
                if done or truncated:reset[:]=True
                elif info.get('reset_agents'):reset[0,info['reset_agents']]=True
                history.push(torch.as_tensor(env.observe(),dtype=torch.float32)[None],reset)
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
