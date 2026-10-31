"""Task boundary, training split and privileged-input regression checks."""

import torch
import pytest

from q1_soccer.data import ExpertReplay
from q1_soccer.algorithm import SoccerSuperMimic
from supermimic.algorithm import SuperMimic


def fake_replay():
    replay = ExpertReplay.__new__(ExpertReplay)
    replay.generator = torch.Generator().manual_seed(1)
    replay.successes = [0]
    replay.episodes = []
    for index in range(2):
        obs = torch.arange(20.0)[:, None].expand(-1, 79).clone() + index * 100
        replay.episodes.append(
            dict(
                actions=torch.full((20, 22), float(index)),
                states=torch.full((21, 109), float(index)),
                obs=obs,
                initial_history=obs[:1].expand(50, -1).clone(),
                context=torch.zeros(20, 62),
                terminated=torch.zeros(20),
                contacts=torch.zeros(20, dtype=torch.bool),
            )
        )
    return replay


def test_vendor_learning_core_is_inherited():
    assert SoccerSuperMimic.fit_model_tensors is SuperMimic.fit_model_tensors
    assert SoccerSuperMimic._update_actor is SuperMimic._update_actor
    assert SoccerSuperMimic._fit_critic is SuperMimic._fit_critic


def test_windows_never_cross_episode_and_imitation_rejects_failed_episode():
    replay = fake_replay()
    for imitation in [True, False]:
        batch = replay.sample(64, window=8, imitation=imitation)
        assert (batch.frames <= 12).all()
        assert torch.equal(
            batch.states[:, :, 0], batch.clips[:, None].expand(-1, 8).float()
        )
        if imitation:
            assert (batch.clips == 0).all()
        assert torch.equal(
            batch.history[:, 0, 0], batch.frames.float() + 100 * batch.clips
        )
        assert torch.equal(
            batch.history[:, 1, 0],
            (batch.frames - 1).clamp_min(0).float() + 100 * batch.clips,
        )


def test_reference_end_pads_only_within_the_same_episode():
    replay = fake_replay()
    sequence = replay.sequence(
        torch.tensor([0, 1]), torch.tensor([18, 19]), torch.arange(15)
    )
    assert sequence.shape == (2, 15, 109)
    assert (sequence[0] == 0).all() and (sequence[1] == 1).all()
    assert torch.equal(
        replay.remaining(torch.tensor([0, 1]), torch.tensor([18, 19])),
        torch.tensor([2, 1]),
    )


def test_original_unbounded_dataset_is_rejected_before_loading_shards(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"status":"complete","control_contract":"original_expert"}'
    )
    with pytest.raises(ValueError, match="bounded-targets"):
        ExpertReplay(tmp_path)


def test_insufficient_gpu_headroom_exits_before_tensor_allocation(monkeypatch):
    from q1_soccer.train import TrainConfig, reserve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * 1024**3, 32 * 1024**3))

    def unexpected(*args):
        pytest.fail(
            "Allocator budget should not be set when another job occupies required VRAM"
        )

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", unexpected)
    with pytest.raises(RuntimeError, match="Insufficient free VRAM"):
        reserve_device(TrainConfig(gpu_memory_gb=3.0, gpu_reserve_gb=3.0))
