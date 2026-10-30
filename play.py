"""Evaluate a trusted soccer checkpoint in native MuJoCo (viewer optional)."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import torch
from q1_supermimic.environment import load_environment
from train import build_agent, validate_checkpoint, CURRICULUM_FIELDS
from q1_supermimic.visualization import draw_camera_frustum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--best', action='store_true', help='Use the best soccer actor with productive-kick evidence')
    p.add_argument('--accepted', action='store_true', help='Use the safety-accepted actor, which may not kick yet')
    p.add_argument('--hide-frustum', action='store_true')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = {**ckpt['config'], 'device': 'cpu'}
    env = load_environment('cpu', 1)
    validate_checkpoint(ckpt, env)
    env.env.restore_near_start_state(ckpt['near_start'])
    for key, value in ckpt['curriculum'].items():
        if key in CURRICULUM_FIELDS:
            setattr(env.env, key, value)
    env.env.set_training_iteration(ckpt['iteration'])
    env.reset()
    agent = build_agent(env.state_dim, cfg)
    agent.load_state_dict(ckpt['agent'])
    if args.best and args.accepted:
        raise ValueError('Choose --best or --accepted')
    if args.best or args.accepted:
        safety = ckpt.get('safety') or {}
        key = 'soccer_best_actor' if args.best else 'accepted_actor'
        if safety.get(key) is None:
            raise ValueError('No soccer best with productive-kick evidence yet; use --accepted to inspect current progress')
        agent.actor.load_state_dict(safety[key])
        limit_key = 'soccer_best_action_limit' if args.best else 'accepted_action_limit'
        if safety.get(limit_key) is not None:
            agent.action_limit.fill_(safety[limit_key])
    agent.eval()
    import mujoco
    render_data = mujoco.MjData(env.env.model)
    context = nullcontext(None)
    visible = [not args.hide_frustum]
    def toggle_frustum(key):
        if key in (ord('F'), ord('f')):
            visible[0] = not visible[0]
    if args.viewer:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(env.env.model, render_data, key_callback=toggle_frustum)
    reward_sum, episodes, falls, successes = 0., 0, 0, 0
    with context as viewer, torch.no_grad():
        if viewer:
            viewer.cam.distance = 5
            viewer.cam.elevation = -20
        state = env.state()
        for _ in range(args.steps):
            began = time.monotonic()
            state, _, reward, info = env.step(agent.act(state))
            reward_sum += reward.item()
            episodes += info['done'].sum().item()
            falls += info['fall'].sum().item()
            successes += info['success'].sum().item()
            if viewer:
                if not viewer.is_running():
                    break
                render_data.qpos[:] = env.env.qpos[0].numpy()
                render_data.qvel[:] = env.env.qvel[0].numpy()
                mujoco.mj_forward(env.env.model, render_data)
                viewer.cam.lookat[:] = render_data.sensor('position').data
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if visible[0]:
                        draw_camera_frustum(viewer.user_scn, env.env.model, render_data, env.cfg)
                viewer.sync()
                time.sleep(max(0, env.env.dt - (time.monotonic() - began)))
    print({'reward': reward_sum, 'completed_episodes': episodes,
           'falls': falls, 'successes': successes})


if __name__ == '__main__':
    main()
