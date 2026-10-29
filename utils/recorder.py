# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import time
import json
import math
from collections import deque
from pathlib import Path
import yaml
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter


def index_checkpoint(path, score=None):
    """Atomically index existing snapshots; aliases do not duplicate weights."""
    path = Path(path)
    metadata = path.parent / 'checkpoint_selection.json'
    state = json.loads(metadata.read_text()) if metadata.exists() else {}
    iteration = int(path.stem.split('_')[-1])

    def link(name):
        target = path.parent / name
        temporary = path.parent / ('.' + name + '.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(path.name)
        os.replace(temporary, target)

    if iteration >= state.get('last_iteration', -1):
        link('last.pth')
        state['last_iteration'] = iteration
    if score is not None and all(math.isfinite(v) for v in score):
        if 'best_score' not in state or tuple(score) > tuple(state['best_score']):
            link('best.pth')
            state.update(best_iteration=iteration, best_score=list(score))
    state['selection_rule'] = 'Among saved snapshots: mean success_rate over the last 100 iterations with completed episodes; tie-break by mean episode reward. Training metrics, not held-out evaluation.'
    temporary = metadata.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, metadata)


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.dir = os.path.join(self.cfg["runner"]["save_path"], "logs", name)
        os.makedirs(self.dir)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        self.wandb_run = None
        self.wandb_metrics = {}
        if self.cfg["runner"]["use_wandb"]:
            self.wandb_run = wandb.init(
                project=self.cfg['runner'].get('wandb_project', self.cfg["basic"]["task"]),
                entity=self.cfg['runner'].get('wandb_entity'),
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )
            self.wandb_run.define_metric('iteration')
            self.wandb_run.define_metric('*', step_metric='iteration')

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None
        self.recent_scores = deque(maxlen=100)

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record and self.last_episode['steps']:
            if self.last_episode['steps']:
                self.recent_scores.append((self._mean(self.last_episode['success_rate']),
                                           self._mean(self.last_episode['reward'])))
            for key in self.last_episode.keys():
                path = ("" if key in ["steps", "reward", "success_rate", "fall_rate"] else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.wandb_run is not None:
                    self.wandb_metrics[path] = value
                self.last_episode[key].clear()
            self.writer.flush()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.wandb_run is not None:
                self.wandb_metrics[key] = float(value)
        if self.wandb_run is not None:
            self.wandb_run.log({**self.wandb_metrics, 'iteration': it + 1}, step=it + 1, commit=True)
            self.wandb_metrics.clear()

    def close(self):
        self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        temporary = path + '.tmp'
        torch.save(model_dict, temporary)
        os.replace(temporary, path)
        score = None
        if self.recent_scores:
            score = tuple(sum(values) / len(self.recent_scores) for values in zip(*self.recent_scores))
        index_checkpoint(path, score)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)


def watch_existing_run(run_dir, poll_seconds=10):
    """Index a pre-upgrade training process without interrupting its optimizer.

New runs index checkpoints directly in save(); this bridge is only needed for
an already-running process that loaded the previous Recorder implementation.
"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run_dir = Path(run_dir)
    events = EventAccumulator(str(run_dir / 'summaries'), size_guidance={'scalars': 0})
    processed = set()
    while True:
        events.Reload()
        tags = events.Tags()['scalars']
        if all(k in tags for k in ('steps', 'success_rate', 'reward')):
            steps = events.Scalars('steps')
            successes = {e.step: e.value for e in events.Scalars('success_rate')}
            rewards = {e.step: e.value for e in events.Scalars('reward')}
            for path in sorted((run_dir / 'nn').glob('model_*.pth'), key=lambda p: int(p.stem.split('_')[-1])):
                iteration = int(path.stem.split('_')[-1])
                if iteration in processed or not steps or steps[-1].step < iteration - 1:
                    continue
                # Old saves were not atomic; only publish fully readable files.
                try:
                    torch.load(path, map_location='cpu', weights_only=True)
                except (EOFError, RuntimeError, OSError):
                    continue
                valid = [e.step for e in steps if e.step < iteration and e.value > 0 and e.step in successes and e.step in rewards][-100:]
                score = (sum(successes[i] for i in valid) / len(valid), sum(rewards[i] for i in valid) / len(valid)) if valid else None
                index_checkpoint(path, score)
                processed.add(iteration)
                print(f'Indexed checkpoint {iteration}, score={score}', flush=True)
        time.sleep(poll_seconds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Index checkpoints from an already-running pre-upgrade trainer.')
    parser.add_argument('--watch-run', type=Path, required=True)
    watch_existing_run(parser.parse_args().watch_run)
