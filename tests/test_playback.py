"""Student playback must use the supplied actor, independently of expert weights."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from expert.control import BoundedSoccerPlayer
from expert.runtime import ROOT as EXPERT_ROOT
from q1_soccer.models import VisionPolicy
from q1_soccer.play import StudentPlayback


def test_standalone_ckpt_preserves_reset_and_drives_bounded_native_physics(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    cfg = yaml.safe_load((EXPERT_ROOT / "configs/Q1.yaml").read_text())
    policy = VisionPolicy(cfg)
    checkpoint = tmp_path / "student.ckpt"
    torch.save(
        {
            "format": "primebot_supermimic_soccer_v1",
            "iteration": 123,
            "actor": policy.state_dict(),
        },
        checkpoint,
    )
    original = BoundedSoccerPlayer(seed=42)
    expected_qpos = original.data.qpos.copy()
    expected_qvel = original.data.qvel.copy()
    load = torch.load
    loaded = []

    def student_only(path, *args, **kwargs):
        assert Path(path) == checkpoint, (
            "Playback attempted to load an extra checkpoint"
        )
        loaded.append(Path(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", student_only)
    playback = StudentPlayback(checkpoint, seed=42)
    np.testing.assert_array_equal(playback.player.data.qpos, expected_qpos)
    np.testing.assert_array_equal(playback.player.data.qvel, expected_qvel)
    assert loaded == [checkpoint]
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(
            playback.player.policy.policy.state_dict()[name], value, rtol=0, atol=0
        )
    sim = playback.run(headless=True, steps=20)
    assert sim.total_steps == 20
    assert np.isfinite(sim.data.qpos).all()
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    assert (sim.targets >= limits[:, 0] - 1e-7).all()
    assert (sim.targets <= limits[:, 1] + 1e-7).all()
    assert (np.abs(sim.data.ctrl) <= sim.effort_limits).all()
    assert all(parameter.device.type == "cpu" for parameter in sim.policy.parameters())


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {}},
        {"format": "primebot_supermimic_soccer_v1"},
    ],
)
def test_wrong_checkpoint_cannot_silently_play_the_expert(tmp_path, payload):
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        StudentPlayback(checkpoint)


def test_passive_viewer_waits_for_its_render_thread_before_returning(monkeypatch):
    import threading
    import expert.play as playback

    closed = threading.Event()
    render_threads = []

    class Handle:
        def close(self):
            closed.set()

    def launch(*args, **kwargs):
        thread = threading.Thread(
            target=closed.wait, name="test (_launch_internal)", daemon=True
        )
        render_threads.append(thread)
        thread.start()
        return Handle()

    monkeypatch.setattr(playback.mujoco.viewer, "launch_passive", launch)
    with playback.managed_viewer(None, None):
        assert render_threads[0].is_alive()
    assert closed.is_set()
    assert not render_threads[0].is_alive()
