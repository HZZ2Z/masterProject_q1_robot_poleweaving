"""Causal history of existing observations, shared by learning and playback."""
import torch
from torch import nn


class HistoryEncoder(nn.Module):
    """Encode five ordered 200 ms averages from the last second of observations."""
    def __init__(self, frame_dim, bins, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(frame_dim * bins, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden))
        # An old actor initially produces exactly the same distribution.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, history):
        return self.layers(history)


class ActorHistory:
    def __init__(self, count, agents, frame_dim, frames, bins, device):
        if frames < 1 or bins < 1 or frames % bins:
            raise ValueError('History frames must be positive and divisible by bins')
        self.frames, self.bins = frames, bins
        self.data = torch.zeros((frames, count, agents, frame_dim), device=device)
        self.cursor = 0

    @torch.no_grad()
    def push(self, obs, reset=None):
        if reset is not None:
            self.data[:, reset] = 0
        self.data[self.cursor].copy_(obs)
        self.cursor = (self.cursor + 1) % self.frames

    @torch.no_grad()
    def pooled(self):
        width = self.frames // self.bins
        result = []
        # Oldest to newest. Quantize identically in sampling and PPO replay.
        for index in range(self.bins):
            slots = (torch.arange(width, device=self.data.device) + self.cursor + index * width) % self.frames
            result.append(self.data[slots].mean(0))
        return torch.stack(result, -2).to(torch.float16)


def history_payload(current, pooled, shared=False, owners=None):
    """Retain current frame fields; only append summaries of previous same fields."""
    if pooled is None:
        return current
    history = pooled.float()
    if shared:
        from .learning import with_team_identity
        history = with_team_identity(history, owners[..., None], role_suffix=6)
    return torch.cat((current, history.flatten(-2)), -1)


def configure_history(args, options, checkpoint=None):
    saved = checkpoint.get('args', {}) if checkpoint else {}
    for key, default in (('history_frames', 0), ('history_bins', 5)):
        value = getattr(options, key, None)
        value = saved.get(key, default) if value is None else value
        setattr(args, key, int(value));setattr(options, key, int(value))
    if args.history_frames < 0 or args.history_bins < 1 or (args.history_frames and args.history_frames % args.history_bins):
        raise ValueError('History frames must be zero or a positive multiple of history bins')
    return bool(checkpoint and saved.get('history_frames', 0) != args.history_frames)
