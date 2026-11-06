"""Synchronous data parallel MAPPO without changing upstream optimizer code."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def enabled():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def moments(x):
    """Population moments weighted by sample count, including unequal shards."""
    values = torch.stack((x.double().sum(), x.double().square().sum(),
                          torch.tensor(x.numel(), device=x.device, dtype=torch.float64)))
    if enabled():
        dist.all_reduce(values)
    mean = values[0] / values[2].clamp(min=1)
    variance = (values[1] / values[2].clamp(min=1) - mean.square()).clamp(min=0)
    return mean.to(x.dtype), variance.to(x.dtype)


class Distributed:
    def __init__(self, device):
        self.device = torch.device(device)
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group('nccl' if self.device.type == 'cuda' else 'gloo',
                                    timeout=timedelta(seconds=180))
        self.primary = self.rank == 0
        self.hooks = []

    def mean(self, value):
        result = value.clone()
        if enabled():
            dist.all_reduce(result)
            result.div_(self.world_size)
        return result

    def maximum(self, value):
        result = torch.as_tensor(value, device=self.device).clone()
        if enabled():
            dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return result.item()

    def objects(self, value):
        if not enabled():
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def check_health(self,check):
        """Reach a shared decision before any rank enters gradient collectives."""
        error=None
        try:check()
        except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        if self.maximum(int(error is not None)):
            errors=self.objects(error)
            details='; '.join(f'rank{rank}: {message}' for rank,message in enumerate(errors) if message)
            raise RuntimeError('Synchronized physics failure: '+details)

    def close(self):
        """Abort outstanding communicator resources on normal AND exceptional exit."""
        for hook in self.hooks:hook.remove()
        self.hooks.clear()
        if dist.is_initialized():dist.destroy_process_group()

    def rows(self, value):
        """Gather variable episode rows, interleaving ranks for fair gate windows."""
        if not enabled():
            return value
        size = torch.tensor([len(value)], device=self.device, dtype=torch.long)
        sizes = [torch.empty_like(size) for _ in range(self.world_size)]
        dist.all_gather(sizes, size)
        lengths = torch.cat(sizes)
        count = int(lengths.max().item())
        if not count:
            return value
        padded = value.new_zeros((count, value.shape[1]))
        padded[:len(value)] = value
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        mask = torch.arange(count, device=self.device)[:, None] < lengths[None, :]
        return torch.stack(gathered, dim=1)[mask]

    def attach(self, policy):
        """Average leaf gradients BEFORE upstream clipping and Adam.step()."""
        if not enabled():
            return
        def average(gradient):
            result = gradient.contiguous().clone()
            dist.all_reduce(result)
            return result.div_(self.world_size)
        for module in (policy.actor, policy.critic):
            for parameter in module.parameters():
                dist.broadcast(parameter.data, src=0)
                if parameter.requires_grad:
                    self.hooks.append(parameter.register_hook(average))

    def check_parameters(self, teams):
        if not enabled():
            return 0.
        parameters = [p.detach().flatten() for policy, _, _ in teams
                      for module in (policy.actor, policy.critic) for p in module.parameters()]
        local = torch.cat(parameters)
        reference = local.clone()
        dist.broadcast(reference, src=0)
        error = self.maximum((local-reference).abs().max())
        if error != 0:
            raise RuntimeError(f'Distributed policy divergence: {error}')
        return error
