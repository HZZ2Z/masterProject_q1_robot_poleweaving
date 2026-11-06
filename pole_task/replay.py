"""Pole replay adapter preserving the Q1 expert's newest-first history order."""

from __future__ import annotations

from supermimic.replay import ReplayBatch, TemporalReplay


class PoleReplay(TemporalReplay):
    def sample(self, batch_size: int, window: int = 8, history: int = 10) -> ReplayBatch:
        batch = super().sample(batch_size, window=window, history=history)
        batch.history = batch.history.flip(dims=(1,))
        return batch
