"""Independent football/style value targets with the existing shared critic trunk."""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.util import huber_loss, mse_loss
from .distributed import enabled


class GroupedValueHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Keep the original parameter names AND objects for exact Adam handoff.
        self.weight, self.bias = original.weight, original.bias
        self.amp_weight = nn.Parameter(torch.zeros_like(original.weight))
        self.amp_bias = nn.Parameter(torch.zeros_like(original.bias))
        self.amp_value = None

    def forward(self, features):
        self.amp_value = F.linear(features, self.amp_weight, self.amp_bias)
        return F.linear(features, self.weight, self.bias)


class GroupedMAPPO(R_MAPPO):
    """Reuse upstream PPO likelihood, clipping and optimizer steps unchanged."""
    def __init__(self, args, policy, device):
        super().__init__(args, policy, device)
        from .gpu_learning import DeviceValueNorm
        self.amp_value_normalizer = DeviceValueNorm(device)
        self.amp_batch = None
        self.last_task_value_loss = self.last_amp_value_loss = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        task_loss = super().cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        old, returns, mask = self.amp_batch
        amp = self.policy.critic.v_out.amp_value
        clipped = old + (amp-old).clamp(-self.clip_param, self.clip_param)
        self.amp_value_normalizer.update(returns, mask)
        target = self.amp_value_normalizer.normalize(returns)
        loss_fn = (lambda x: huber_loss(x, self.huber_delta)) if self._use_huber_loss else mse_loss
        loss = loss_fn(target-amp)
        if self._use_clipped_value_loss:
            loss = torch.maximum(loss, loss_fn(target-clipped))
        # All ranks execute this reduction, including ranks with no match worlds.
        count = mask.sum().detach().clone()
        if enabled():
            dist.all_reduce(count)
            count /= dist.get_world_size()
        amp_loss = (loss*mask).sum()/count.clamp(min=1)
        self.last_task_value_loss, self.last_amp_value_loss = task_loss.detach(), amp_loss.detach()
        return task_loss + amp_loss
