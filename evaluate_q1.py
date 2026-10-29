"""Fixed Q1 soccer benchmark on nominal native MuJoCo, independent of training resets."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from play_mujoco import SoccerPlayer


def fixed_cases():
    return [dict(distance=distance, heading_degrees=heading, ball_xy=ball, seed=1000 + index)
            for index, (distance, heading, ball) in enumerate(
                (d, h, b) for d in (.4, .8, 1.5, 3.) for h in (-60., 0., 60.)
                for b in ((-1., 0.), (1., 1.)))]


def place_case(player, case):
    player.rng = np.random.default_rng(case['seed'])
    player.reset_robot()
    ball = np.array(case['ball_xy'])
    direction = np.array([7., 0.]) - ball
    direction /= np.linalg.norm(direction)
    yaw = np.arctan2(direction[1], direction[0]) + np.deg2rad(case['heading_degrees'])
    a = player.root_addr
    player.data.qpos[a:a + 7] = [*(ball - case['distance'] * direction),
                                player.cfg['init_state']['pos'][2], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    player.data.qpos[player.ball_addr:player.ball_addr + 7] = [*ball, .11, 1, 0, 0, 0]
    player.data.qvel[:] = 0
    player.ball_history[:] = 0
    mujoco.mj_forward(player.model, player.data)


def foot_gap(player):
    ball = player.data.qpos[player.ball_addr:player.ball_addr + 3]
    bounds = np.array(player.cfg['asset']['foot_collision_bounds'])
    gaps = []
    for name in player.cfg['asset']['foot_names']:
        body = player.model.body(name).id
        local = (ball - player.data.xpos[body]) @ player.data.xmat[body].reshape(3, 3)
        gaps.append(max(0., np.linalg.norm(local - np.clip(local, *bounds)) - player.cfg['rewards']['ball_radius']))
    return min(gaps)


def has_foot_contact(player, foot_geoms, ball_geom):
    return any(contact.dist <= 0 and
               ((contact.geom[0] == ball_geom and contact.geom[1] in foot_geoms) or
                (contact.geom[1] == ball_geom and contact.geom[0] in foot_geoms))
               for contact in player.data.contact)


def evaluate(checkpoint, duration=20.):
    torch.set_num_threads(2)
    player = SoccerPlayer(str(checkpoint), reset_mode='default', task='Q1')
    foot_bodies = {player.model.body(n).id for n in player.cfg['asset']['foot_names']}
    foot_geoms = {g for g in range(player.model.ngeom) if player.model.geom_bodyid[g] in foot_bodies and player.model.geom_contype[g]}
    ball_geom = player.model.geom('soccer_ball').id
    results = []
    for case in fixed_cases():
        place_case(player, case)
        initial_distance = np.linalg.norm(np.array(case['ball_xy']) - [7., 0.])
        reached = touched = False
        first_touch_s = None
        max_progress = post_touch_progress = 0.
        progress_at_touch = 0.
        near_stall_s = near_idle_s = 0.
        last_ball = player.data.qpos[player.ball_addr:player.ball_addr + 2].copy()
        cause = None
        for step in range(round(duration / player.model.opt.timestep)):
            cause = player.step(auto_reset=False)
            if cause:
                break
            ball = player.data.qpos[player.ball_addr:player.ball_addr + 2]
            progress = initial_distance - np.linalg.norm(ball - [7., 0.])
            max_progress = max(max_progress, progress)
            if has_foot_contact(player, foot_geoms, ball_geom) and not touched:
                touched = True
                first_touch_s = (step + 1) * player.model.opt.timestep
                progress_at_touch = progress
            if touched:
                post_touch_progress = max(post_touch_progress, progress - progress_at_touch)
            if player.physics_steps % player.decimation == 0:
                gap = foot_gap(player)
                reached |= gap < .12  # frozen benchmark threshold, independent of curriculum
                direction = np.array([7., 0.]) - last_ball
                speed = np.dot(ball - last_ball, direction / max(np.linalg.norm(direction), 1e-6)) / (player.decimation * player.model.opt.timestep)
                near_idle_s = near_idle_s + player.decimation * player.model.opt.timestep if gap < .12 and speed <= .05 else 0.
                if near_idle_s > 1.5:
                    near_stall_s += player.decimation * player.model.opt.timestep
                last_ball = ball.copy()
        results.append(dict(**case, reached=bool(reached), touched=bool(touched), goal=cause == 'goal',
                            fall=cause == 'fall', terminal=cause or 'evaluation_horizon', first_touch_s=first_touch_s,
                            max_goal_progress_m=max_progress, post_touch_progress_m=post_touch_progress,
                            effective_kick=bool(touched and post_touch_progress >= .2),
                            post_touch_fall=bool(touched and cause == 'fall'), near_stall_s=near_stall_s))
        print(f"case {len(results)}/24: touch={touched}, progress={post_touch_progress:.2f}m, end={cause}", flush=True)
    mean = lambda key: sum(row[key] for row in results) / len(results)
    touches = sum(row['touched'] for row in results)
    summary = {key: mean(key) for key in ('reached', 'touched', 'goal', 'fall', 'effective_kick', 'post_touch_progress_m', 'near_stall_s')}
    summary['post_touch_fall_rate'] = sum(row['post_touch_fall'] for row in results) / touches if touches else None
    by_distance = {}
    for distance in (.4, .8, 1.5, 3.):
        group = [row for row in results if row['distance'] == distance]
        by_distance[str(distance)] = {key: sum(row[key] for row in group) / len(group)
                                     for key in ('reached', 'touched', 'goal', 'fall', 'effective_kick')}
    return dict(suite='q1_fixed_v1', physics='native_mujoco_nominal', duration_s=duration,
                camera_fov=player.cfg['asset']['camera_fov'], camera_rot=player.cfg['asset']['camera_rot'],
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                summary=summary, by_distance=by_distance, cases=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.duration)
    output = args.output or Path('logs/q1/evaluations') / (Path(args.checkpoint).stem + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(output)
