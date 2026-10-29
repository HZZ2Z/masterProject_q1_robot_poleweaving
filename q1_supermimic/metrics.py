"""Completed-episode metric names/units shared with the PPO dashboard."""
from collections import deque
import torch


class EpisodeMetrics:
    def __init__(self):
        self.names = None
        self.window = deque(maxlen=1000)

    def update(self, values, done):
        if self.names is None:
            self.names = list(values)
            matrix = torch.stack(list(values.values()), -1)
            self.pending = torch.zeros_like(matrix)
            self.completed = matrix.new_zeros(matrix.shape[-1])
            self.count = matrix.new_zeros(())
        matrix = torch.stack([values[k] for k in self.names], -1)
        self.pending += matrix
        self.completed += self.pending[done].sum(0)
        self.window.extend(self.pending[done].detach().cpu().tolist())
        self.count += done.sum()
        self.pending[done] = 0

    def drain(self, dt):
        if self.names is None:
            return {}
        packed = torch.cat((self.count[None], self.completed)).detach().cpu().tolist()
        self.completed.zero_()
        self.count.zero_()
        if packed[0] == 0:
            return {}
        means = dict(zip(self.names, [v / packed[0] for v in packed[1:]]))
        result = {k if k in ('reward', 'steps', 'success_rate', 'fall_rate') else f'episode/{k}': v
                  for k, v in means.items()}
        result['episode/duration_s'] = means['steps'] * dt
        instant = {f'instant/{k}': v for k, v in result.items()}
        rows = torch.tensor(list(self.window), dtype=torch.float64)
        rolling = dict(zip(self.names, rows.mean(0).tolist()))
        result = {k if k in ('reward', 'steps', 'success_rate', 'fall_rate') else f'episode/{k}': v
                  for k, v in rolling.items()}
        result.update(instant)
        result['episode/duration_s'] = rolling['steps'] * dt
        result['episode/fall_rate'] = rolling['fall_rate']
        result['episode/success_rate'] = rolling['success_rate']
        result['statistics/window_episodes'] = len(self.window)
        result['statistics/window_capacity'] = self.window.maxlen
        result['statistics/window_ready'] = float(len(self.window) == self.window.maxlen)
        result['statistics/new_completed_episodes'] = packed[0]
        result['statistics/window_simulated_seconds'] = rows[:, self.names.index('steps')].sum().item() * dt
        return result
