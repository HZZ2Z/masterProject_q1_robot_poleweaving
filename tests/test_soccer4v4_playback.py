import importlib.util
from pathlib import Path
from types import SimpleNamespace


PLAY_PATH = Path(__file__).parents[1] / "courses/soccer-4v4/play.py"
SPEC = importlib.util.spec_from_file_location("soccer_4v4_play", PLAY_PATH)
playback = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(playback)


def test_old_stage_checkpoint_does_not_require_match_task():
    env = SimpleNamespace(env=SimpleNamespace())

    assert playback.apply_checkpoint_motion_constraints(env, {}) is False


def test_match_checkpoint_restores_motion_constraints():
    motion = SimpleNamespace()
    env = SimpleNamespace(env=SimpleNamespace(match_task=SimpleNamespace(motion=motion)))

    assert playback.apply_checkpoint_motion_constraints(
        env,
        {
            "iteration": 12,
            "motion_constraints": {"enabled": True, "start_iteration": 7},
        },
    ) is True
    assert motion.enabled is True
    assert motion.start_iteration == 7
    assert motion.iteration == 12
