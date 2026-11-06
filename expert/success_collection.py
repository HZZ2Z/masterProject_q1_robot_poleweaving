"""Successful-only expert collection and physical removal of rejected episodes."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import datetime
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .runtime import ROOT
from .control import BoundedSoccerPlayer
from .collect import atomic_json, collect_episode, digest, episode_seed, split_for
from .dataset import validate_arrays


def successful(meta):
    return (
        meta.get("termination") == "goal"
        and meta.get("verified_success") is True
        and meta.get("task_success") is True
        and meta.get("policy_imitation_eligible") is True
    )


class EpisodeStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.rows = [
            json.loads(line)
            for line in (self.directory / "index.jsonl").read_text().splitlines()
        ]
        self.next_id = (
            max((int(row["episode_id"].split("_")[-1]) for row in self.rows), default=0)
            + 1
        )
        self.seeds = {row["seed"] for row in self.rows}

    def summarize(self, **changes):
        self.manifest.update(
            completed_episodes=len(self.rows),
            total_transitions=sum(r["steps"] for r in self.rows),
            policy_imitation_episodes=sum(
                r["policy_imitation_eligible"] for r in self.rows
            ),
            outcomes=dict(Counter(r["termination"] for r in self.rows)),
            splits=dict(Counter(r["split"] for r in self.rows)),
            **changes,
        )
        atomic_json(self.directory / "manifest.json", self.manifest)

    @staticmethod
    def delete_episode(directory, identifier, cache_dirs):
        directory = Path(directory)
        path = directory / (identifier + ".npz")
        removed_bytes = 0
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                for current in [False, True]:
                    identity = hashlib.sha256(b"q1-native-sensor-cache-v1")
                    if current:
                        identity.update(
                            (
                                ROOT / "resources/q1_description/q1_soccer.xml"
                            ).read_bytes()
                        )
                        identity.update((ROOT / "configs/Q1.yaml").read_bytes())
                    for key in ["qpos", "qvel", "qacc_warmstart", "applied_torques"]:
                        identity.update(arrays[key].tobytes())
                    for cache in cache_dirs:
                        cached = Path(cache) / (identity.hexdigest() + ".npy")
                        if cached.exists():
                            removed_bytes += cached.stat().st_size
                            cached.unlink()
        for suffix in [".npz", ".json"]:
            file = directory / (identifier + suffix)
            if file.exists():
                removed_bytes += file.stat().st_size
                file.unlink()
        return removed_bytes

    def purge(self):
        retained = []
        removed = []
        for row in self.rows:
            meta = json.loads(
                (self.directory / (row["episode_id"] + ".json")).read_text()
            )
            (retained if successful(meta) else removed).append(row)
        temporary = self.directory / "index.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in retained))
        temporary.replace(self.directory / "index.jsonl")
        removed_bytes = 0
        for row in removed:
            removed_bytes += self.delete_episode(
                self.directory, row["episode_id"], [self.directory / "sensor_cache"]
            )
        self.rows = retained
        self.summarize(
            status="complete",
            selection_rule="strict_success_only",
            removed_unsuccessful_episodes=len(removed),
        )
        for name in [
            "dataset_check.json",
            "teacher_action_check.json",
            "physics_replay_check.json",
        ]:
            (self.directory / name).unlink(missing_ok=True)
        return {
            "directory": str(self.directory),
            "deleted_episodes": len(removed),
            "retained_successes": len(retained),
            "deleted_bytes": removed_bytes,
        }

    def add(self, arrays, meta, checkpoint_hash):
        if not successful(meta):
            raise ValueError("Unsuccessful episode cannot be persisted")
        if meta["seed"] in self.seeds:
            raise ValueError("Duplicate expert seed")
        validate_arrays(arrays, meta)
        identifier = f"episode_{self.next_id:06d}"
        self.next_id += 1
        meta.update(episode_id=identifier, checkpoint_sha256=checkpoint_hash)
        path = self.directory / (identifier + ".npz")
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        meta["trajectory_sha256"] = digest(path)
        atomic_json(self.directory / (identifier + ".json"), meta)
        keys = [
            "episode_id",
            "seed",
            "split",
            "spawn_group",
            "termination",
            "task_success",
            "verified_success",
            "policy_imitation_eligible",
            "effective_kick",
            "contact",
            "timeout",
            "failure",
            "steps",
            "duration_s",
            "ball_goal_net_progress_m",
            "post_touch_max_progress_m",
            "trajectory_sha256",
        ]
        row = {key: meta[key] for key in keys}
        with (self.directory / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.rows.append(row)
        self.seeds.add(meta["seed"])


_TEACHER = None
_RESET_MODE = "reference"
_CHECKPOINT_HASH = None


def initialize_worker(checkpoint, reset_mode, threads):
    global _TEACHER, _RESET_MODE, _CHECKPOINT_HASH
    torch.set_num_threads(threads)
    _RESET_MODE = reset_mode
    _CHECKPOINT_HASH = digest(checkpoint)
    with contextlib.redirect_stdout(io.StringIO()):
        player = BoundedSoccerPlayer(checkpoint, seed=0, reset_mode=reset_mode)
    _TEACHER = player.policy.eval().requires_grad_(False)


def attempt_episode(seed, group):
    with contextlib.redirect_stdout(io.StringIO()):
        player = BoundedSoccerPlayer(seed=seed, reset_mode=_RESET_MODE, policy=_TEACHER)
    if player.spawn_kind != group:
        raise AssertionError("Seed cohort mismatch")
    initial_rng = json.loads(json.dumps(player.rng.bit_generator.state))
    split = split_for(seed, _CHECKPOINT_HASH)  # fixed before observing the outcome
    arrays, meta = collect_episode(player)
    meta.update(
        seed=seed,
        split=split,
        control_contract=player.control_contract,
        initial_sensor_rng=initial_rng,
        final_sensor_rng=player.rng.bit_generator.state,
        reset_mode=_RESET_MODE,
        joint_names=[player.model.actuator(i).name for i in range(player.model.nu)],
        initial_numpy_seed=seed,
        torch_seed=seed,
    )
    if not successful(meta):
        return None, meta
    return arrays, meta


class SuccessCollector:
    def __init__(
        self,
        directory,
        checkpoint,
        successes,
        seed=50000000,
        workers=8,
        threads=1,
        reset_mode="reference",
    ):
        self.store = EpisodeStore(directory)
        self.checkpoint = Path(checkpoint)
        self.target = successes
        self.seed = seed
        self.workers = workers
        self.threads = threads
        self.reset_mode = reset_mode
        self.stop = False
        self.checkpoint_hash = digest(self.checkpoint)
        if (
            self.store.manifest.get("control_contract")
            != "joint_target_and_torque_limits_v1"
        ):
            raise ValueError("Collection requires bounded-target dataset")
        if self.store.manifest["checkpoint_sha256"] != self.checkpoint_hash:
            raise ValueError("Cannot mix expert checkpoints")
        if any(not successful(row) for row in self.store.rows):
            raise ValueError("Purge unsuccessful episodes before appending")

    def run(self):
        start_count = len(self.store.rows)
        started = time.perf_counter()
        quotas = [
            self.target // 4,
            self.target // 4,
            self.target - 2 * (self.target // 4),
        ]
        counts = [0, 0, 0]
        pending_counts = [0, 0, 0]
        pending = {}
        outcomes = Counter()
        attempts = 0
        cursor = 0
        session = {
            "started_at": datetime.datetime.now().astimezone().isoformat(),
            "target_new_successes": self.target,
            "initial_successes": start_count,
            "workers": self.workers,
            "threads_per_worker": self.threads,
            "seed_start": self.seed,
            "implementation_sha256": {
                name: digest(ROOT / name)
                for name in [
                    "runtime.py",
                    "control.py",
                    "policy.py",
                    "collect.py",
                    "success_collection.py",
                ]
            },
        }
        self.store.manifest.setdefault("collection_sessions", []).append(session)
        self.store.manifest.update(
            planned_episodes=start_count + self.target,
            selection_rule="strict_success_only",
            world_model_rule="strict successful episodes only",
            collection_target_new_successes=self.target,
        )
        self.store.summarize(status="collecting", new_successes=0, attempts=0)

        def stop(*_):
            self.stop = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(str(self.checkpoint), self.reset_mode, self.threads),
            ) as pool:
                while sum(counts) < self.target and not self.stop:
                    while len(pending) < self.workers:
                        group = None
                        for _ in range(4):
                            candidate = [0, 1, 2, 2][cursor % 4]
                            cursor += 1
                            if (
                                counts[candidate] + pending_counts[candidate]
                                < quotas[candidate]
                            ):
                                group = candidate
                                break
                        if group is None:
                            break
                        seed = episode_seed(self.seed, group)
                        self.seed = seed + 1
                        if seed in self.store.seeds:
                            continue
                        future = pool.submit(attempt_episode, seed, group)
                        pending[future] = group
                        pending_counts[group] += 1
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        group = pending.pop(future)
                        pending_counts[group] -= 1
                        arrays, meta = future.result()
                        attempts += 1
                        outcomes[meta["termination"]] += 1
                        if arrays is not None:
                            self.store.add(arrays, meta, self.checkpoint_hash)
                            counts[group] += 1
                        self.store.summarize(
                            status="collecting",
                            new_successes=sum(counts),
                            attempts=attempts,
                            discarded_attempts=attempts - sum(counts),
                            attempt_outcomes=dict(outcomes),
                            new_successes_by_spawn=counts,
                        )
                        if sum(counts) % 25 == 0 or sum(counts) == self.target:
                            print(
                                json.dumps(
                                    {
                                        "new_successes": sum(counts),
                                        "target": self.target,
                                        "attempts": attempts,
                                        "discarded": attempts - sum(counts),
                                        "total_successes": len(self.store.rows),
                                        "elapsed_s": round(
                                            time.perf_counter() - started, 1
                                        ),
                                    }
                                ),
                                flush=True,
                            )
                    if attempts > max(100, self.target * 20):
                        raise RuntimeError(
                            "Too many failed attempts; expert needs investigation"
                        )
                for future in pending:
                    future.cancel()
        except BaseException:
            self.store.summarize(
                status="interrupted", new_successes=sum(counts), attempts=attempts
            )
            raise
        self.store.summarize(
            status="complete" if sum(counts) == self.target else "interrupted",
            new_successes=sum(counts),
            attempts=attempts,
            discarded_attempts=attempts - sum(counts),
            elapsed_seconds=time.perf_counter() - started,
            new_successes_by_spawn=counts,
        )
        if sum(counts) != self.target:
            raise RuntimeError(
                f"Only {sum(counts)} / {self.target} successes collected"
            )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "new_successes": sum(counts),
                    "total_successes": len(self.store.rows),
                    "attempts": attempts,
                }
            ),
            flush=True,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/best.pth"
    )
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=50000000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.successes, args.workers, args.threads) < 1:
        parser.error("Counts must be positive")
    SuccessCollector(
        args.output,
        args.checkpoint,
        args.successes,
        args.seed,
        args.workers,
        args.threads,
    ).run()


if __name__ == "__main__":
    main()
