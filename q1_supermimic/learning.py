"""Autoregressive dynamics fitting and differentiable imagined actor-critic.

Soccer adaptation: learn reward/termination from real contact transitions rather
than use the upstream reference-motion reward. No PPO or AMP update is used.
"""
import torch
from torch import nn
import torch.nn.functional as F
from supermimic.models import mlp, frozen


class TemporalReplay:
    def __init__(self, steps, envs, dim, device):
        self.capacity, self.count, self.cursor = steps, 0, 0
        self.states = torch.empty(steps, envs, dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(steps, envs, 22, device=device)
        self.rewards = torch.empty(steps, envs, device=device)
        self.done = torch.empty(steps, envs, device=device)
        self.fall = torch.empty_like(self.done)

    def add(self, state, action, target, reward, done, fall):
        for buffer, value in zip((self.states, self.actions, self.next_states,
                                  self.rewards, self.done, self.fall),
                                 (state, action, target, reward, done, fall)):
            buffer[self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def sample(self, batch, length, validation=False):
        if self.count < length:
            raise ValueError('Not enough consecutive transitions')
        device = self.states.device
        start = torch.randint(self.count - length + 1, (batch,), device=device)
        oldest = self.cursor if self.count == self.capacity else 0
        times = (oldest + start[:, None] + torch.arange(length, device=device)) % self.capacity
        count = self.states.shape[1]
        held_out = max(1, count // 8) if count >= 8 else 0
        low, high = (count - held_out, count) if validation and held_out else (0, count - held_out)
        worlds = torch.randint(low, high, (batch, 1), device=device)
        return tuple(x[times, worlds] for x in (self.states, self.actions,
                     self.next_states, self.rewards, self.done, self.fall))


class SoccerAgent(nn.Module):
    def __init__(self, dim, obs_dim=79, width=256, horizon=4, lr=5e-5):
        super().__init__()
        self.dim, self.obs_dim, self.horizon = dim, obs_dim, horizon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('action_limit', torch.tensor(.10))
        self.register_buffer('dynamic_mask', torch.ones(dim))
        self.layout = None
        self.last_diagnostics = {}
        self.dynamics = mlp(dim + 22, (width, width, width), dim)
        self.outcomes = mlp(dim + 22 + dim, (width, width), 2)
        self.actor = mlp(obs_dim, (width, width), 22)
        self.critic = mlp(dim, (width, width), 1)
        nn.init.normal_(self.dynamics[-1].weight, std=.001)
        nn.init.zeros_(self.dynamics[-1].bias)
        # Begin near the validated neutral stance, with exploration in real rollouts.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.outcomes[-1].bias[1] = -4.
        self.lr = lr
        self.model_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def make_optimizers(self):
        self.model_optimizer = torch.optim.Adam(
            list(self.dynamics.parameters()) + list(self.outcomes.parameters()), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas=(.9, .95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    @torch.no_grad()
    def calibrate(self, states):
        self.mean.copy_(states.mean(0))
        self.scale.copy_(states.std(0, unbiased=False).clamp_min(.1))

    def normalize(self, state):
        return ((state - self.mean) / self.scale).clamp(-20, 20)

    def act(self, state):
        return self.action_limit * self.actor(self.normalize(state)[..., :self.obs_dim]).tanh()

    def predict(self, state, action):
        following = state + self.dynamic_mask * self.dynamics(torch.cat((state, action), -1))
        if self.layout is not None:
            # Previous-action observation is deterministic, not a free model output.
            raw = following * self.scale + self.mean
            raw = raw.clone()
            raw[..., 57:79] = action
            raw[..., 136:158] = action
            q = self.layout['root_quat']
            quat = raw[..., q:q+4].clone()
            raw[..., q:q+4] = F.normalize(quat, dim=-1)
            following = (raw - self.mean) / self.scale
        outcomes = self.outcomes(torch.cat((state, action, following), -1))
        return following, outcomes[..., 0], outcomes[..., 1]

    @staticmethod
    def optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite learning loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()

    def fit_model(self, batch):
        states, actions, targets, rewards, dones, falls = batch
        targets = self.normalize(targets)
        prediction = self.normalize(states[:, 0])
        alive = torch.ones_like(rewards[:, 0])
        total = prediction.new_zeros(())
        mass = prediction.new_zeros(())
        for t in range(actions.shape[1]):
            prediction, reward, terminal = self.predict(prediction, actions[:, t])
            error = (prediction - targets[:, t]).square()
            loss = error.mean(-1)
            if self.layout is not None:
                loss = loss + error[:, self.layout['physical_start']:self.layout['physical_end']].mean(-1)
            loss = loss + F.smooth_l1_loss(reward, rewards[:, t] * .1, reduction='none')
            loss = loss + F.binary_cross_entropy_with_logits(terminal, falls[:, t], reduction='none')
            total = total + (alive * loss).sum()
            mass = mass + alive.sum()
            # Never fit across either a timeout or a physical termination/reset.
            alive = alive * (1 - dones[:, t])
        loss = total / mass.clamp_min(1)
        self.optimize(loss, self.model_optimizer,
                      list(self.dynamics.parameters()) + list(self.outcomes.parameters()))
        return loss.item()

    def configure_task(self, layout):
        self.layout = layout
        # Keep domain parameters and curriculum context fixed in imagination.
        self.dynamic_mask.fill_(0)
        for start, end in layout['dynamic_ranges']:
            self.dynamic_mask[start:end] = 1

    @torch.no_grad()
    def validate_model(self, batch):
        states, actions, targets, _, dones, _ = batch
        initial = self.normalize(states[:, 0])
        predicted = initial
        total = initial.new_zeros(5)
        mass = initial.new_zeros(())
        alive = initial.new_ones(len(initial))
        a, b = (self.layout['physical_start'], self.layout['physical_end']) if self.layout else (0, self.dim)
        for t in range(actions.shape[1]):
            predicted = self.predict(predicted, actions[:, t])[0]
            target = self.normalize(targets[:, t])
            error = (predicted[:, a:b]-target[:, a:b]).square().mean(-1)
            persistence = (initial[:, a:b]-target[:, a:b]).square().mean(-1)
            if self.layout:
                raw = predicted*self.scale+self.mean
                l = self.layout
                j, v = l['joints_start'], l['root_velocity']
                height = (raw[:, l['root_z']]-targets[:, t, l['root_z']]).abs()
                joints = (raw[:, j:j+22]-targets[:, t, j:j+22]).square().mean(-1)
                velocity = (raw[:, v:v+6]-targets[:, t, v:v+6]).square().mean(-1)
            else:
                height = joints = velocity = error
            total += torch.stack([(x*alive).sum() for x in (error, persistence, height, joints, velocity)])
            mass += alive.sum()
            alive *= 1-dones[:, t]
        values = total/mass.clamp_min(1)
        return {'model_validation/physical_mse': values[0].item(),
                'model_validation/persistence_ratio': (values[0]/values[1].clamp_min(1e-6)).item(),
                'model_validation/height_mae_m': values[2].item(),
                'model_validation/joint_rmse_rad': values[3].sqrt().item(),
                'model_validation/root_velocity_rmse': values[4].sqrt().item()}

    def fit_critic(self, batch):
        states, _, targets, rewards, _, falls = batch
        states, targets = self.normalize(states), self.normalize(targets)
        with torch.no_grad():
            # Targets come from real transitions, including pre-reset timeout targets.
            target = rewards * .1 + .99 * (1 - falls) * self.critic(targets).squeeze(-1)
        loss = F.smooth_l1_loss(self.critic(states).squeeze(-1), target)
        self.optimize(loss, self.critic_optimizer, self.critic.parameters())
        return loss.item()

    def physical_reward(self, current, following, action):
        # Explicit smooth planning objective; the actual complete v4 reward is
        # still used for real critic TD targets and for acceptance evaluation.
        if self.layout is None:
            return -.01 * following.square().mean(-1) - action.square().sum(-1), following.new_ones(len(following))
        c, n = current * self.scale + self.mean, following * self.scale + self.mean
        l = self.layout
        q = l['root_quat']
        w, x, y, z = n[:, q:q+4].unbind(-1)
        upright = (2 * (x*z - w*y)).square() + (2 * (y*z + w*x)).square()
        height = n[:, l['root_z']]
        vel = l['root_velocity']
        speed2 = n[:, vel:vel+6].square().sum(-1)
        continuation = torch.sigmoid((height - .2) / .015) * torch.sigmoid((50 - speed2) / 3)
        joints = n[:, l['joints_start']:l['joints_start']+22]
        defaults = n[:, l['default_start']:l['default_start']+22]
        progress = c[:, l['progress_index']].clamp(0, 1)
        dt = l['dt']
        stall = torch.maximum(c[:, l['progress_index']+4], c[:, l['progress_index']+5]).clamp(0, 1)
        productive_survival = 3 * (1 - progress * stall) if l.get('productive_survival', False) else 3
        reward = (productive_survival - 50 * (height - l['nominal_height']).square() - 5 * upright
                  - (1-progress) * (joints-defaults).square().mean(-1)) * dt
        reward = reward - dt * (action-c[:, 57:79]).square().sum(-1)
        reward = reward - .001 / dt * (n[:, vel:vel+6]-c[:, vel:vel+6]).square().sum(-1)
        ball = l['ball_start']
        bp, bn = c[:, ball:ball+2], n[:, ball:ball+2]
        root = l['root_xy']
        approach = (bn-c[:, root:root+2]).norm(dim=-1) - (bn-n[:, root:root+2]).norm(dim=-1)
        goal = bn.new_tensor([7., 0.])
        forward = (bp-goal).norm(dim=-1) - (bn-goal).norm(dim=-1)
        approach_gain = 1 + c[:, -1].clamp(0, 1) * (l.get('near_approach_gain', 1.) - 1)
        reward = reward + progress * dt * (50 * approach_gain * approach + 500 * forward)
        if l.get('productive_survival', False):
            near = c[:, l['progress_index']+3].clamp(0, 1)
            stall_cost = near * 2 * c[:, l['progress_index']+4] + (1-near) * c[:, l['progress_index']+5]
            reward = reward - progress * dt * l.get('stall_penalty_factor', 1.) * stall_cost
        # Known termination cost, not the learned termination head's gradient.
        reward = reward - 20 * (1-continuation)
        return reward * .1, continuation

    def improve(self, real_states, real_batch=None, exploration_std=.01):
        state = self.normalize(real_states).detach()
        before = [p.detach().clone() for p in self.actor.parameters()]
        with torch.no_grad():
            reference_action = self.action_limit * self.actor(state[:, :self.obs_dim]).tanh()
        with frozen(self.dynamics, self.outcomes, self.critic):
            alive = state.new_ones(len(state))
            objective = state.new_zeros(len(state))
            for t in range(self.horizon):
                action = self.action_limit * self.actor(state[:, :self.obs_dim]).tanh()
                following, _, _ = self.predict(state, action)
                reward, continuation = self.physical_reward(state, following, action)
                objective = objective + (.99 ** t) * alive * (reward - .01 * action.square().sum(-1))
                alive = alive * continuation
                state = following
            objective = objective + (.99 ** self.horizon) * alive * self.critic(state).squeeze(-1)
            current = self.normalize(real_states).detach()
            proposed = self.action_limit * self.actor(current[:, :self.obs_dim]).tanh()
            planning_loss = -objective.mean()
            actor_loss = planning_loss + 10 * (proposed-reference_action).square().mean()
            if real_batch is not None:
                real, executed, targets, rewards, _, falls = real_batch
                real, targets = self.normalize(real), self.normalize(targets)
                with torch.no_grad():
                    advantage = rewards * .1 + .99 * (1-falls) * self.critic(targets).squeeze(-1) - self.critic(real).squeeze(-1)
                    advantage = ((advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-5)).clamp(-5,5)
                    squashed = (executed/self.action_limit).clamp(-.999999,.999999)
                    latent = torch.atanh(squashed)
                mean = self.actor(real[:, :self.obs_dim])
                std = exploration_std / float(self.action_limit)
                log_prob = torch.distributions.Normal(mean, std).log_prob(latent).sum(-1)
                real_loss = -(log_prob * advantage).mean()
                # The model remains an auxiliary proposal; real transitions
                # provide the dominant score-function policy gradient. No PPO.
                actor_loss = real_loss + .01 * planning_loss + .001 * mean.square().mean()
                self.last_diagnostics = {'loss/real_policy': real_loss.item(),
                                         'loss/planning_policy': planning_loss.item()}

            self.optimize(actor_loss, self.actor_optimizer, self.actor.parameters())
        # Bound one optimizer update on real observations. This complements the
        # global action bound and the separate real-simulator acceptance gate.
        with torch.no_grad():
            delta = (self.act(real_states)-reference_action).abs().max()
            if delta > .005:
                alpha = float(.005 / delta)
                for p, old in zip(self.actor.parameters(), before):
                    p.copy_(old + alpha * (p-old))
        return actor_loss.item(), 0.
