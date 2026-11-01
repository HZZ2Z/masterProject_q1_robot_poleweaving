"""Foot geometry, productive strikes and bounded inactivity feedback for soccer."""
import weakref
import torch


class MatchBall:
    terms = ('foot_approach', 'near_ball_stall', 'far_ball_stall',
             'forward_strike', 'side_strike', 'goal_progress')
    metrics = tuple(f'{key}_{team}' for key in
                    ('foot_gap', 'near_fraction', 'near_stall', 'far_stall', 'left_strike', 'right_strike')
                    for team in ('red', 'blue'))

    def __init__(self, env):
        self.env = weakref.proxy(env)
        n, d = env.count, env.device
        self.near = torch.zeros((n, 8), device=d, dtype=torch.bool)
        self.near_idle = torch.zeros((n, 8), device=d)
        self.far_idle = torch.zeros_like(self.near_idle)
        self.near_stall = torch.zeros_like(self.near_idle)
        self.far_stall = torch.zeros_like(self.near_idle)
        self.window_steps = torch.zeros((n, 8), device=d, dtype=torch.long)
        self.anchor_gap = torch.zeros_like(self.near_idle)
        self.anchor_ball = torch.zeros((n, 8, 3), device=d)
        self.goal_speed = torch.zeros_like(self.near_idle)
        self.effective = torch.zeros((n, 8), device=d, dtype=torch.bool)
        self.legal_owner = torch.full((n,), -1, device=d, dtype=torch.long)
        self.best_goal_distance = torch.zeros((n, 2), device=d)
        self.chase = torch.zeros_like(self.near)
        if hasattr(env, 'reference'):
            ref = env.reference.data
            root = int(env.bodies[0])
            rotation = env.tensor(ref.xmat[root].reshape(3, 3))
            feet = env.feet[0].cpu().numpy()
            self.neutral_feet = env.tensor(ref.xpos[feet] - ref.xpos[root]) @ rotation
            foot_rotation = env.tensor(ref.xmat[feet].reshape(2, 3, 3))
            self.neutral_foot_rotation = torch.einsum('ij,fjk->fik', rotation.T, foot_rotation)

    def gaps(self, ball=None, feet=None, rotation=None):
        e = self.env
        feet = e.xpos[:, e.feet] if feet is None else feet
        rotation = e.xmat[:, e.feet].reshape(e.count, 8, 2, 3, 3) if rotation is None else rotation
        ball = e.ball[:, None, None, :] if ball is None else ball
        relative = torch.einsum('nafi,nafij->nafj', ball - feet, rotation)
        closest = relative.clamp(min=e.foot_bounds[0], max=e.foot_bounds[1])
        return (torch.linalg.vector_norm(relative - closest, dim=-1) - e.ball_radius).clamp(min=0)

    def distances_to_goals(self):
        e = self.env
        goals = e.ball.new_tensor([[7., 0.], [-7., 0.]])
        return torch.linalg.vector_norm(e.ball[:, None, :2] - goals, dim=-1)

    def reset_worlds(self, ids):
        for name in ('near', 'near_idle', 'far_idle', 'near_stall', 'far_stall',
                     'window_steps', 'goal_speed', 'effective', 'chase'):
            getattr(self, name)[ids] = 0
        self.legal_owner[ids] = -1
        self.best_goal_distance[ids] = self.distances_to_goals()[ids]

    def reset_agents(self, w, a):
        # The rally's strike history and progress budget survive individual falls.
        for name in ('near', 'near_idle', 'far_idle', 'near_stall', 'far_stall',
                     'window_steps', 'goal_speed', 'chase'):
            getattr(self, name)[w, a] = 0

    def step(self, before, foot_before, chase, healthy, foot, body, foot_valid, rotation_before=None):
        e = self.env
        feet = e.xpos[:, e.feet]
        rotation = e.xmat[:, e.feet].reshape(e.count, 8, 2, 3, 3)
        rotation_before = rotation if rotation_before is None else rotation_before
        gap = self.gaps().amin(-1)
        self.chase[:] = chase
        was_near = self.near.clone()
        self.near[:] = (gap < .12 + .04 * was_near.float()) & chase
        goals = torch.stack((e.sign * 7, torch.zeros_like(e.sign)), -1)
        direction = goals[None] - before[:, None, :2]
        direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
        self.goal_speed[:] = ((e.ball - before)[:, None, :2] * direction).sum(-1) / e.dt
        team_contacts = foot.reshape(e.count, 2, 4).any(-1)
        unique = team_contacts.sum(-1) == 1
        owner = team_contacts.long().argmax(-1)
        self.legal_owner[:] = torch.where(body, -1, torch.where(
            team_contacts.any(-1), torch.where(unique, owner, -1), self.legal_owner))
        productive = (self.goal_speed > .05) & foot_valid[:, None]
        self.near_idle[:] = torch.where(self.near & ~productive, self.near_idle + e.dt, 0.)
        self.near_stall[:] = ((self.near_idle - 1.5) / 1.5).clamp(0, 1)

        # Measure complete windows against a fixed ball. Ball-only movement and
        # foot oscillation cannot repeatedly clear the far inactivity timer.
        start = self.window_steps == 0
        self.anchor_ball[start] = before[:, None].expand(-1, 8, -1)[start]
        self.anchor_gap[start] = self.gaps(before[:, None, None], foot_before, rotation_before).amin(-1)[start]
        self.window_steps += 1
        finished = self.window_steps >= max(1, round(2. / e.dt))
        remaining = self.gaps(self.anchor_ball[:, :, None], feet, rotation).amin(-1)
        approach = self.anchor_gap - remaining >= .04
        self.far_idle[:] = torch.where(self.near | ~chase | (finished & approach), 0.,
                                      self.far_idle + finished.float() * 2.)
        self.window_steps[finished | ~chase] = 0
        pending = self.window_steps * e.dt
        self.far_stall[:] = torch.where(chase & ~self.near,
                                       ((self.far_idle + pending - 2.) / 1.).clamp(0, 1), 0.)

        fixed_ball = before[:, None, None]
        previous_gap = self.gaps(fixed_ball, foot_before, rotation_before).amin(-1)
        current_gap = self.gaps(fixed_ball, feet, rotation).amin(-1)
        delta = (previous_gap - current_gap).clamp(-.05, .05)
        close = torch.linalg.vector_norm(e.xpos[:, e.bodies, :2] - e.ball[:, None, :2], dim=-1) < .8
        foot_progress = 6. * (delta.clamp(min=0) * healthy + delta.clamp(max=0)) * chase * close

        velocity = (feet - foot_before) / e.dt
        local = torch.einsum('nafi,nafij->nafj', velocity, rotation)
        forward, side = local[..., 0], local[..., 1]
        magnitude = forward.abs() + side.abs()
        valid = (e.foot_hits.bool() & foot[..., None] & ~body[:, None, None] &
                 (magnitude > .05))
        # Share a control step's ball motion across contacting feet; two feet
        # and diagonal strikes cannot collect duplicate full rewards.
        credit = valid.float() / valid.sum(-1, keepdim=True).clamp(min=1)
        speed = self.goal_speed.clamp(0, 2)[..., None]
        straight = (credit * speed * forward.clamp(min=0) / magnitude.clamp(min=1e-6)).sum(-1)
        lateral = (credit * speed * side.abs() / magnitude.clamp(min=1e-6)).sum(-1)
        strikes = valid & (self.goal_speed > .2)[..., None]
        self.effective |= strikes.any(-1)

        distance = self.distances_to_goals()
        novel = (self.best_goal_distance - distance).clamp(0, .04)
        teams = torch.arange(2, device=e.device)[None]
        legal = (self.legal_owner[:, None] == teams) & foot_valid[:, None] & ~body[:, None]
        progress = 4. * novel * legal
        # Consume even illegal advancement, so a later foot touch cannot claim
        # body-pushed distance. Never restore this budget on a robot reset.
        self.best_goal_distance[:] = torch.minimum(self.best_goal_distance, distance)
        rewards = dict(foot_approach=foot_progress,
                       near_ball_stall=-e.dt * .8 * self.near_stall,
                       far_ball_stall=-e.dt * .35 * self.far_stall,
                       forward_strike=e.dt * 2. * straight,
                       side_strike=e.dt * 8. * lateral,
                       goal_progress=progress.repeat_interleave(4, -1))
        def chaser_mean(value):
            return (value * chase).reshape(e.count, 2, 4).sum(-1).mean(0)
        metrics = torch.cat((chaser_mean(gap), chaser_mean(self.near.float()),
                             chaser_mean((self.near_stall > 0).float()),
                             chaser_mean((self.far_stall > 0).float()),
                             strikes[..., 0].float().reshape(e.count, 2, 4).mean((0, 2)),
                             strikes[..., 1].float().reshape(e.count, 2, 4).mean((0, 2))))
        return rewards, metrics

    def place_near(self, ids, robots):
        """Neutral left/right foot practice with a genuine 10--25 cm surface gap."""
        if not len(ids):
            return
        e = self.env
        ball = e.ball[ids]
        goal = torch.stack((e.sign[robots] * 7, torch.zeros_like(e.sign[robots])), -1)
        heading = torch.atan2(goal[:, 1] - ball[:, 1], goal[:, 0] - ball[:, 0])
        yaw = heading + e.uniform((len(ids),), -torch.pi / 6, torch.pi / 6)
        c, s = yaw.cos(), yaw.sin()
        z = torch.zeros_like(c);o = torch.ones_like(c)
        frame = torch.stack((c,-s,z,s,c,z,z,z,o), -1).reshape(-1, 3, 3)
        foot = (e.uniform((len(ids),)) > .5).long()
        offset = torch.einsum('nij,nj->ni', frame, self.neutral_feet[foot])
        foot_frame = frame @ self.neutral_foot_rotation[foot]
        gap = e.uniform((len(ids),), .10, .25)
        vertical = (ball[:, 2] - .417 - offset[:, 2] - e.foot_bounds[1, 2]).clamp(min=0)
        horizontal = ((gap + e.ball_radius).square() - vertical.square()).clamp(min=0).sqrt()
        local = torch.zeros_like(ball);local[:, 0] = e.foot_bounds[1, 0] + horizontal
        world = torch.einsum('nij,nj->ni', foot_frame, local)
        e.qpos[ids, e.rootq[robots]] = ball[:, 0] - offset[:, 0] - world[:, 0]
        e.qpos[ids, e.rootq[robots] + 1] = ball[:, 1] - offset[:, 1] - world[:, 1]
        e.qpos[ids, e.rootq[robots] + 3] = (yaw / 2).cos()
        e.qpos[ids, e.rootq[robots] + 6] = (yaw / 2).sin()
