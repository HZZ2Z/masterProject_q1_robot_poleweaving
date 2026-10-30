"""GPU-resident rollout/GAE using the official MAPPO ppo_update implementation."""
import torch
from torch import nn


class DeviceValueNorm(nn.Module):
    """Registered persistent statistics; denormalization never copies to CPU."""
    def __init__(self, device):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(1, device=device))
        self.register_buffer('running_mean_sq', torch.zeros(1, device=device))
        self.register_buffer('debiasing_term', torch.zeros((), device=device))

    def running_mean_var(self):
        weight = self.debiasing_term.clamp(min=1e-5)
        mean = self.running_mean/weight
        return mean, (self.running_mean_sq/weight-mean.square()).clamp(min=.01)

    @torch.no_grad()
    def update(self, x):
        self.running_mean.mul_(.99999).add_(x.mean(0)*.00001)
        self.running_mean_sq.mul_(.99999).add_(x.square().mean(0)*.00001)
        self.debiasing_term.mul_(.99999).add_(.00001)

    def normalize(self, x):
        mean, var = self.running_mean_var()
        return (x-mean)/var.sqrt()

    def denormalize(self, x):
        mean, var = self.running_mean_var()
        return x*var.sqrt()+mean


class GpuRollout:
    def __init__(self, horizon, count, device, obs_dim=121):
        self.horizon, self.count, self.device = horizon, count, device
        zeros = lambda *shape: torch.zeros(shape, device=device)
        # Central critic inputs reconstructed per minibatch: don't store 8 copies of global state.
        self.obs = zeros(horizon+1, count, 8, obs_dim)
        self.actions = zeros(horizon, count, 8, 22)
        self.log_probs = zeros(horizon, count, 8, 1)
        self.values = zeros(horizon+1, count, 8, 1)  # unnormalized at collection time
        self.rewards = zeros(horizon, count, 8, 1)
        self.masks = torch.ones((horizon, count, 8, 1), device=device)
        self.returns = zeros(horizon, count, 8, 1)

    def compute_returns(self, gamma=.99, lam=.95):
        gae = torch.zeros_like(self.values[0])
        for step in reversed(range(self.horizon)):
            delta = self.rewards[step]+gamma*self.values[step+1]*self.masks[step]-self.values[step]
            gae = delta+gamma*lam*self.masks[step]*gae
            self.returns[step] = gae+self.values[step]

    def update(self, team, trainer, args, active=None):
        sl = slice(team*4, team*4+4)
        adv = self.returns[:, :, sl]-self.values[:-1, :, sl]
        valid = torch.ones_like(adv,dtype=torch.bool) if active is None else active[:,sl][None,:,:,None].expand_as(adv)
        selected_adv=adv[valid]
        adv = (adv-selected_adv.mean())/(selected_adv.std(unbiased=False)+1e-5)
        norm = trainer.value_normalizer
        # Snapshot normalized old values before ppo_update changes ValueNorm statistics.
        old_values = norm.normalize(self.values[:-1, :, sl]).detach()
        total = self.horizon*self.count*4
        sums = torch.zeros(8, device=self.device)
        updates = 0
        for _ in range(args.ppo_epoch):
            eligible = valid.flatten().nonzero().flatten()
            for index in eligible[torch.randperm(eligible.numel(),device=self.device)].tensor_split(args.num_mini_batch):
                t = index//(self.count*4)
                w = index//4 % self.count
                a = index % 4
                global_obs = self.obs[t, w]
                order = torch.arange(8, device=self.device).roll(-team*4)
                state = torch.cat((global_obs[:, order].flatten(1), torch.eye(4, device=self.device)[a]), -1)
                obs = global_obs[torch.arange(index.numel(), device=self.device), a+team*4]
                rnn = torch.zeros((index.numel(), 1, args.hidden_size), device=self.device)
                mask = torch.ones((index.numel(), 1), device=self.device)
                sample = (state, obs, rnn, rnn, self.actions[t, w, a+team*4], old_values[t, w, a],
                          self.returns[t, w, a+team*4], mask, mask, self.log_probs[t, w, a+team*4],
                          adv[t, w, a], None)
                result = trainer.ppo_update(sample)
                iw=result[-1].detach()
                approximate_kl=((iw-1)-iw.clamp(min=1e-12).log()).mean()
                clip_fraction=((iw-1).abs()>trainer.clip_param).float().mean()
                sums += torch.stack([x.detach().mean() for x in result]+[approximate_kl,clip_fraction])
                updates += 1
        names = ('value_loss', 'critic_grad_norm', 'policy_loss', 'dist_entropy', 'actor_grad_norm', 'ratio', 'approx_kl', 'clip_fraction')
        return dict(zip(names, (sums/updates).cpu().tolist()))
