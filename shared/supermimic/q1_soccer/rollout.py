"""Bounded native MuJoCo rollout, shared by evaluation and online replay."""

from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import multiprocessing

import torch

from expert.control import BoundedSoccerPlayer
from expert.collect import collect_episode
from .models import RuntimePolicy

_WORKER_POLICY = None


def _initialize(policy):
    global _WORKER_POLICY
    torch.set_num_threads(1)
    _WORKER_POLICY = RuntimePolicy(policy)


def _episode(task):
    seed, max_steps = task
    with contextlib.redirect_stdout(io.StringIO()):
        player = BoundedSoccerPlayer(seed=seed, reset_mode="reference")
    player.policy = _WORKER_POLICY
    arrays, meta = collect_episode(player, max_control_steps=max_steps)
    meta.update(seed=seed, split="train", control_contract=player.control_contract)
    return arrays, meta


class NativeRollouts:
    def __init__(self, workers=2):
        self.workers = workers

    def run(self, policy, seeds, max_steps=None):
        cpu_policy = RuntimePolicy(policy).policy
        tasks = [(seed, max_steps) for seed in seeds]
        if self.workers == 1:
            _initialize(cpu_policy)
            return [_episode(task) for task in tasks]
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(cpu_policy,),
        ) as pool:
            return list(pool.map(_episode, tasks))

    def evaluate(self, policy, seeds, max_steps=None):
        episodes = self.run(policy, seeds, max_steps)
        total = len(episodes)
        return dict(
            episodes=total,
            verified_success_rate=sum(meta["verified_success"] for _, meta in episodes)
            / total,
            goal_rate=sum(meta["task_success"] for _, meta in episodes) / total,
            fall_rate=sum(meta["termination"] == "fall" for _, meta in episodes)
            / total,
            timeout_rate=sum(meta["termination"] == "timeout" for _, meta in episodes)
            / total,
            out_rate=sum(meta["termination"] == "out" for _, meta in episodes) / total,
            collector_limit_rate=sum(
                meta["termination"] == "collector_limit" for _, meta in episodes
            )
            / total,
            mean_post_touch_progress=sum(
                meta["post_touch_max_progress_m"] for _, meta in episodes
            )
            / total,
            seeds=list(seeds),
        )
