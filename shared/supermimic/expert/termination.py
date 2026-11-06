"""Original playback terrain bounds; no training environment imports."""
import torch


class FlatTerrain:
    border_size = 20.
    env_length = 20.
    env_width = 20.

    def terrain_heights(self, xy):
        return torch.zeros(xy.shape[:-1], dtype=xy.dtype, device=xy.device)
