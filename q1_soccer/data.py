"""Episode-disjoint expert replay and successful motion references."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from expert.dataset import validate_arrays
from expert.runtime import ROOT as EXPERT_ROOT
from supermimic.replay import ReplayBatch
from .robot import physical_states
from .sensors import native_sensors

CONTROL_CONTRACT = "joint_target_and_torque_limits_v1"


class ExpertReplay:
    dt = 0.02

    @staticmethod
    def is_success(meta):
        return (
            meta.get("termination") == "goal"
            and meta.get("verified_success") is True
            and meta.get("task_success") is True
            and meta.get("policy_imitation_eligible") is True
        )

    def __init__(self, directory, *, split="train", seed=0):
        self.directory = Path(directory)
        self.successful_only = split == "train"
        self.excluded_unsuccessful = 0
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["status"] != "complete":
            raise ValueError("Expert collection is incomplete")
        if self.manifest.get("control_contract") != CONTROL_CONTRACT:
            raise ValueError(
                "Recollect with expert.collect --bounded-targets; original unbounded transitions are incompatible"
            )
        for field, relative in [
            ("config_sha256", "configs/Q1.yaml"),
            ("scene_sha256", "resources/q1_description/q1_soccer.xml"),
        ]:
            if (
                hashlib.sha256((EXPERT_ROOT / relative).read_bytes()).hexdigest()
                != self.manifest[field]
            ):
                raise ValueError(
                    f"Current expert configuration/model differs from dataset: {relative}"
                )
        bundle = json.loads((self.directory / "source_bundle.json").read_text())
        for item in bundle["files"]:
            if item["target"].startswith(("resources/", "data/")):
                if (
                    hashlib.sha256(
                        (EXPERT_ROOT / item["target"]).read_bytes()
                    ).hexdigest()
                    != item["sha256"]
                ):
                    raise ValueError(
                        f"Expert asset differs from dataset: {item['target']}"
                    )
        self.episodes = []
        self.generator = torch.Generator().manual_seed(seed)
        for row in map(
            json.loads, (self.directory / "index.jsonl").read_text().splitlines()
        ):
            if row["split"] != split:
                continue
            identifier = row["episode_id"]
            path = self.directory / (identifier + ".npz")
            meta = json.loads((self.directory / (identifier + ".json")).read_text())
            if self.successful_only and not self.is_success(meta):
                self.excluded_unsuccessful += 1
                continue
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != meta["trajectory_sha256"]
            ):
                raise ValueError(f"Trajectory hash mismatch: {identifier}")
            with np.load(path, allow_pickle=False) as shard:
                arrays = {name: shard[name] for name in shard.files}
            validate_arrays(arrays, meta)
            if not len(arrays["actions"]):
                continue
            context = np.concatenate(
                (
                    arrays["ball_observation_queue"].reshape(-1, 60),
                    np.full((meta["steps"], 1), meta["ball_delay_steps"]),
                    (np.arange(meta["steps"]) % 2)[:, None],
                ),
                -1,
            )
            sensors = native_sensors(arrays, self.directory / "sensor_cache")
            self.episodes.append(
                dict(
                    meta=meta,
                    states=physical_states(arrays, sensors),
                    actions=torch.tensor(arrays["actions"]),
                    obs=torch.tensor(arrays["observations"]),
                    initial_history=torch.tensor(arrays["initial_history"]),
                    context=torch.tensor(context, dtype=torch.float32),
                    terminated=torch.tensor(arrays["terminated"], dtype=torch.float32),
                    contacts=torch.tensor(arrays["foot_ball_contact"].any(-1)),
                )
            )
        if not self.episodes:
            raise ValueError(f"No episodes in {split} split")
        self.successes = [
            i
            for i, episode in enumerate(self.episodes)
            if episode["meta"]["policy_imitation_eligible"]
        ]
        if not self.successes:
            raise ValueError(f"No verified successful references in {split}")

    def history(self, episode, frames):
        # Negative indexes address reset padding rather than another episode.
        indexes = frames[:, None] - torch.arange(50)[None]
        observed = episode["obs"][indexes.clamp_min(0)]
        padding = episode["initial_history"][(-indexes).clamp(0, 49)]
        return torch.where((indexes >= 0)[..., None], observed, padding)

    def sample(self, batch_size, window=8, *, imitation=False, device="cpu"):
        candidates = self.successes if imitation else list(range(len(self.episodes)))
        counts = torch.tensor(
            [max(0, len(self.episodes[i]["actions"]) - window + 1) for i in candidates]
        )
        if not counts.sum():
            raise ValueError("No complete temporal windows")
        selected = torch.multinomial(
            counts.float(), batch_size, replacement=True, generator=self.generator
        )
        output = {
            key: []
            for key in [
                "states",
                "next_states",
                "actions",
                "context",
                "terminated",
                "history",
                "clips",
                "frames",
            ]
        }
        for choice in selected.unique():
            slots = (selected == choice).nonzero().flatten()
            index = candidates[int(choice)]
            episode = self.episodes[index]
            frames = torch.randint(
                int(counts[choice]), (len(slots),), generator=self.generator
            )
            # Half the sampled starts are close to real foot/ball contact when
            # available. The rest preserve approach, recovery and failure states.
            contacts = episode["contacts"].nonzero().flatten()
            if len(contacts):
                mask = torch.rand(len(frames), generator=self.generator) < 0.5
                count = int(mask.sum())
                chosen = contacts[
                    torch.randint(len(contacts), (count,), generator=self.generator)
                ]
                frames[mask] = (
                    chosen - torch.randint(25, (count,), generator=self.generator)
                ).clamp(0, int(counts[choice]) - 1)
            steps = frames[:, None] + torch.arange(window)
            values = dict(
                states=episode["states"][steps],
                next_states=episode["states"][steps + 1],
                actions=episode["actions"][steps],
                context=episode["context"][frames],
                terminated=episode["terminated"][steps],
                history=self.history(episode, frames),
                clips=torch.full_like(frames, index),
                frames=frames,
            )
            for key, value in values.items():
                output[key].append(value)
        # Group order is arbitrary but identical for all fields.
        return ReplayBatch(
            **{key: torch.cat(value).to(device) for key, value in output.items()}
        )

    def sequence(self, clips, frames, offsets):
        device = clips.device
        indexes = (frames[:, None] + offsets[None]).cpu()
        rows = [
            self.episodes[int(clip)]["states"][
                row.clamp_max(len(self.episodes[int(clip)]["actions"]))
            ]
            for clip, row in zip(clips.cpu(), indexes)
        ]
        return torch.stack(rows).to(device)

    def remaining(self, clips, frames):
        lengths = [len(self.episodes[int(clip)]["actions"]) for clip in clips.cpu()]
        return torch.tensor(lengths, device=frames.device) - frames

    def add(self, arrays, meta):
        """Online training episodes share this same contract and boundary logic."""
        if not self.is_success(meta):
            return False
        validate_arrays(arrays, meta)
        if not meta["steps"]:
            return False
        n = meta["steps"]
        context = np.concatenate(
            (
                arrays["ball_observation_queue"].reshape(n, 60),
                np.full((n, 1), meta["ball_delay_steps"]),
                (np.arange(n) % 2)[:, None],
            ),
            -1,
        )
        sensors = native_sensors(arrays, self.directory / "sensor_cache")
        self.episodes.append(
            dict(
                meta=meta,
                states=physical_states(arrays, sensors),
                actions=torch.tensor(arrays["actions"]),
                obs=torch.tensor(arrays["observations"]),
                initial_history=torch.tensor(arrays["initial_history"]),
                context=torch.tensor(context, dtype=torch.float32),
                terminated=torch.tensor(arrays["terminated"], dtype=torch.float32),
                contacts=torch.tensor(arrays["foot_ball_contact"].any(-1)),
            )
        )
        # Online successes are dynamics data. Expert references remain frozen.
        return True
