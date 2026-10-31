"""Rejected trajectories must be removed physically, not merely hidden by masks."""

import json

import numpy as np
import pytest

from expert.success_collection import EpisodeStore, successful


def test_purge_unlinks_failed_shards_and_rebuilds_index_and_manifest(tmp_path):
    rows = []
    for index, cause in enumerate(["goal", "timeout"], 1):
        identifier = f"episode_{index:06d}"
        success = cause == "goal"
        row = dict(
            episode_id=identifier,
            seed=index,
            split="train",
            steps=1,
            termination=cause,
            policy_imitation_eligible=success,
            verified_success=success,
            task_success=success,
        )
        rows.append(row)
        (tmp_path / (identifier + ".json")).write_text(json.dumps(row))
        np.savez(
            tmp_path / (identifier + ".npz"),
            qpos=np.zeros((2, 36)),
            qvel=np.zeros((2, 34)),
            qacc_warmstart=np.zeros((2, 34)),
            applied_torques=np.zeros((1, 10, 22)),
        )
    (tmp_path / "manifest.json").write_text('{"status":"complete"}')
    (tmp_path / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = EpisodeStore(tmp_path).purge()
    assert result["deleted_episodes"] == 1 and result["retained_successes"] == 1
    assert (tmp_path / "episode_000001.npz").exists()
    assert not (tmp_path / "episode_000002.npz").exists()
    assert not (tmp_path / "episode_000002.json").exists()
    assert [
        json.loads(line)["episode_id"]
        for line in (tmp_path / "index.jsonl").read_text().splitlines()
    ] == ["episode_000001"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["completed_episodes"] == 1 and manifest["outcomes"] == {"goal": 1}
    with pytest.raises(ValueError, match="Unsuccessful"):
        EpisodeStore(tmp_path).add({}, rows[1], "unused")


def test_goal_without_verified_kick_is_not_a_success():
    assert not successful(
        {
            "termination": "goal",
            "task_success": True,
            "verified_success": False,
            "policy_imitation_eligible": False,
        }
    )
