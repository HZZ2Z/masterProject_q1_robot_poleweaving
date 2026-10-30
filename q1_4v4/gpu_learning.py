"""GPU-resident rollout/GAE using the official MAPPO ppo_update implementation."""
import torch
import torch.distributed as dist
from torch import nn
from .distributed import enabled, moments
from .learning import with_team_identity


def normalize_by_course(adv,valid,course_ids):
    """Global moments per task prevent reward scale from dominating other courses."""
    ids=course_ids[None,:,None,None].expand_as(adv)
    selected=adv[valid].double(); labels=ids[valid]
    stats=torch.zeros((28,3),device=adv.device,dtype=torch.float64)
    stats[:,0].scatter_add_(0,labels,selected)
    stats[:,1].scatter_add_(0,labels,selected.square())
    stats[:,2].scatter_add_(0,labels,torch.ones_like(selected))
    if enabled():dist.all_reduce(stats)
    mean=stats[:,0]/stats[:,2].clamp(min=1)
    variance=(stats[:,1]/stats[:,2].clamp(min=1)-mean.square()).clamp(min=0)
    center=mean.to(adv.dtype)[course_ids][None,:,None,None]
    scale=variance.to(adv.dtype)[course_ids][None,:,None,None].sqrt()+1e-5
    return (adv-center)/scale


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
        if enabled():
            mean, variance = moments(x)
            second = variance + mean.square()
        else:
            mean, second = x.mean(0), x.square().mean(0)
        self.running_mean.mul_(.99999).add_(mean*.00001)
        self.running_mean_sq.mul_(.99999).add_(second*.00001)
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
        self.course_ids=None
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
        shared = team is None
        agents = 8 if shared else 4
        sl = slice(None) if shared else slice(team*4, team*4+4)
        adv = self.returns[:, :, sl]-self.values[:-1, :, sl]
        valid = torch.ones_like(adv,dtype=torch.bool) if active is None else active[:,sl][None,:,:,None].expand_as(adv)
        selected_adv=adv[valid]
        if self.course_ids is not None:
            adv=normalize_by_course(adv,valid,self.course_ids)
        elif enabled():
            mean, variance = moments(selected_adv)
            adv = (adv-mean)/(variance.sqrt()+1e-5)
        else:
            adv = (adv-selected_adv.mean())/(selected_adv.std(unbiased=False)+1e-5)
        norm = trainer.value_normalizer
        # Snapshot normalized old values before ppo_update changes ValueNorm statistics.
        old_values = norm.normalize(self.values[:-1, :, sl]).detach()
        sums = torch.zeros(8, device=self.device)
        updates = 0
        for _ in range(args.ppo_epoch):
            eligible = valid.flatten().nonzero().flatten()
            # Keep shared minibatches at the former per-team size to bound VRAM.
            batches = args.num_mini_batch*(2 if shared else 1)
            for index in eligible[torch.randperm(eligible.numel(),device=self.device)].tensor_split(batches):
                t = index//(self.count*agents)
                w = index//agents % self.count
                a = index % agents
                global_obs = self.obs[t, w]
                owner = a//4 if shared else torch.full_like(a,team)
                robot = a if shared else a+team*4
                order = (torch.arange(8,device=self.device)[None]+owner[:,None]*4)%8
                state = torch.cat((global_obs.gather(1,order[:,:,None].expand(-1,-1,global_obs.shape[-1])).flatten(1),
                                   torch.eye(4, device=self.device)[a%4]), -1)
                obs = global_obs[torch.arange(index.numel(), device=self.device), robot]
                if shared:
                    obs = with_team_identity(obs,owner)
                    state = with_team_identity(state,owner,role_suffix=4)
                rnn = torch.zeros((index.numel(), 1, args.hidden_size), device=self.device)
                mask = torch.ones((index.numel(), 1), device=self.device)
                sample = (state, obs, rnn, rnn, self.actions[t, w, robot], old_values[t, w, a],
                          self.returns[t, w, robot], mask, mask, self.log_probs[t, w, robot],
                          adv[t, w, a], None)
                result = trainer.ppo_update(sample)
                iw=result[-1].detach()
                approximate_kl=((iw-1)-iw.clamp(min=1e-12).log()).mean()
                clip_fraction=((iw-1).abs()>trainer.clip_param).float().mean()
                sums += torch.stack([x.detach().mean() for x in result]+[approximate_kl,clip_fraction])
                updates += 1
        names = ('value_loss', 'critic_grad_norm', 'policy_loss', 'dist_entropy', 'actor_grad_norm', 'ratio', 'approx_kl', 'clip_fraction')
        return dict(zip(names, (sums/updates).cpu().tolist()))
