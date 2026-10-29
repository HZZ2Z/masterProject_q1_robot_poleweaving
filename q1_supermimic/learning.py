"""Supervised autoregressive dynamics and real-trajectory GAE policy learning."""
import torch
from torch import nn
import torch.nn.functional as F
from supermimic.models import mlp


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
        self.dim, self.obs_dim, self.validation_horizon = dim, obs_dim, horizon
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
        # Keep domain parameters and curriculum context fixed in model rollouts.
        self.dynamic_mask.fill_(0)
        for start, end in layout['dynamic_ranges']:
            self.dynamic_mask[start:end] = 1

    @torch.no_grad()
    def validate_model(self, batch):
        states, actions, targets, _, dones, _ = batch
        if actions.shape[1] > self.validation_horizon:
            raise ValueError('Validation sequence exceeds configured horizon')
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

    def learn_real_rollout(self, rollout, exploration_std, gamma=.995, lam=.98,
                           critic_epochs=4, max_kl=.005, update_actor=True):
        """One on-policy GAE update; replay/model gradients never enter the actor.

        rollout is time-major. Latent Gaussian samples are stored before tanh;
        done stops GAE traces, while terminated alone disables bootstrapping.
        """
        import copy
        states, _, targets, rewards, dones, terminated, latents = rollout
        shape = rewards.shape
        flat = states.flatten(0, 1)
        normalized = self.normalize(flat).detach()
        with torch.no_grad():
            values = self.critic(normalized).squeeze(-1).reshape(shape)
            next_values = self.critic(self.normalize(targets.flatten(0, 1))).squeeze(-1).reshape(shape)
            advantages, returns = real_gae(rewards * .1, values, next_values, dones, terminated, gamma, lam)
            ids = torch.randperm(len(flat), device=flat.device)[:8192]
            x = normalized[ids]
            target = returns.flatten()[ids].detach()
            advantage = advantages.flatten()[ids]
            advantage_std = advantage.std(unbiased=False)
            advantage = (advantage - advantage.mean()) / advantage_std.clamp_min(1e-6)
            sampled_latent = latents.flatten(0, 1)[ids].detach()
            obs = x[:, :self.obs_dim]
            old_mean = self.actor(obs).detach()
            std = exploration_std / float(self.action_limit)
            old_logp = torch.distributions.Normal(old_mean, std).log_prob(sampled_latent).sum(-1)
            variance = target.var(unbiased=False).clamp_min(1e-8)
            before_mse = (self.critic(x).squeeze(-1)-target).square().mean()
        # Targets are detached ONCE before any optimizer step; no stale replay.
        critic_losses = []
        for _ in range(critic_epochs):
            order = torch.randperm(len(x), device=x.device)
            for batch in order.split(2048):
                loss = F.mse_loss(self.critic(x[batch]).squeeze(-1), target[batch])
                self.optimize(loss, self.critic_optimizer, self.critic.parameters())
                critic_losses.append(loss.item())
        with torch.no_grad():
            fitted = self.critic(x).squeeze(-1)
            after_mse = (fitted-target).square().mean()
            explained = 1 - (fitted-target).var(unbiased=False) / variance
        loss_value, kl, alpha = 0., 0., 0.
        if update_actor and advantage_std > 1e-6:
            parameters = list(self.actor.parameters())
            before = [p.detach().clone() for p in parameters]
            optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
            mean = self.actor(obs)
            logp = torch.distributions.Normal(mean, std).log_prob(sampled_latent).sum(-1)
            loss = -(logp * advantage).mean()
            loss_value = loss.item()
            self.optimize(loss, self.actor_optimizer, parameters)
            proposed = [p.detach().clone() for p in parameters]
            alpha = 1.
            with torch.no_grad():
                for _ in range(12):
                    for p, old, new in zip(parameters, before, proposed):
                        p.copy_(old + alpha * (new-old))
                    new_mean = self.actor(obs)
                    kl = ((new_mean-old_mean).square() / (2*std*std)).sum(-1).mean().item()
                    logp = torch.distributions.Normal(new_mean, std).log_prob(sampled_latent).sum(-1)
                    improvement = ((logp-old_logp).clamp(-20,20).exp() * advantage).mean().item()
                    if kl <= max_kl and improvement >= -1e-6:
                        break
                    alpha *= .5
                else:
                    for p, old in zip(parameters, before):
                        p.copy_(old)
                    alpha, kl = 0., 0.
                if alpha < 1:
                    self.actor_optimizer.load_state_dict(optimizer_before)
        self.last_diagnostics = {
            'learning/policy_kl': kl, 'learning/update_fraction': alpha,
            'learning/value_mse_before': before_mse.item(),
            'learning/value_mse_after': after_mse.item(),
            'learning/explained_variance': explained.item(),
        }
        return loss_value, sum(critic_losses) / max(1, len(critic_losses))


def real_gae(rewards, values, next_values, dones, terminated, gamma=.995, lam=.98):
    """Separate bootstrap and trace masks; never propagate through auto-reset."""
    advantages = torch.zeros_like(rewards)
    future = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1-terminated[t]) * next_values[t] - values[t]
        future = delta + gamma * lam * (1-dones[t]) * future
        advantages[t] = future
    return advantages.detach(), (advantages + values).detach()
