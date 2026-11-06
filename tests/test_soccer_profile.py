"""Regression checks for the unchanged soccer architecture and checkpoint path.

The soccer profile owns the teacher-initialized ``VisionPolicy``/``Soccer*``
networks and the ``primebot_supermimic_soccer_v1`` checkpoint format.  These
tests pin the pre-profile-migration state-dict contract, round-trip a
checkpoint written by the same ``save`` function used before the migration,
and prove that loading it needs no pole-only settings.
"""

from pathlib import Path
from types import SimpleNamespace

import torch

from q1_soccer.models import SoccerCritic, SoccerWorldModel, VisionPolicy
from q1_soccer.profiles import resolve_profile
from q1_soccer.robot import SoccerRobot

# Recorded from the soccer models at the baseline commit before the pole
# profile migration; no pole-only tensor may enter these networks.
SOCCER_STATE_DICT_SHAPES = {
    "actor": {
        "encoder.0.weight": [1024, 3950],
        "encoder.0.bias": [1024],
        "encoder.2.weight": [128, 1024],
        "encoder.2.bias": [128],
        "encoder.4.weight": [64, 128],
        "encoder.4.bias": [64],
        "decoder.0.weight": [256, 143],
        "decoder.0.bias": [256],
        "decoder.2.weight": [256, 256],
        "decoder.2.bias": [256],
        "decoder.4.weight": [128, 256],
        "decoder.4.bias": [128],
        "decoder.6.weight": [22, 128],
        "decoder.6.bias": [22],
    },
    "model": {
        "scale": [109],
        "neutral": [22],
        "action_scale": [22],
        "limits": [22, 2],
        "dynamics.0.weight": [512, 131],
        "dynamics.0.bias": [512],
        "dynamics.2.weight": [512, 512],
        "dynamics.2.bias": [512],
        "dynamics.4.weight": [512, 512],
        "dynamics.4.bias": [512],
        "dynamics.6.weight": [87, 512],
        "dynamics.6.bias": [87],
        "termination.0.weight": [128, 109],
        "termination.0.bias": [128],
        "termination.2.weight": [64, 128],
        "termination.2.bias": [64],
        "termination.4.weight": [1, 64],
        "termination.4.bias": [1],
    },
    "critic": {
        "scale": [109],
        "network.0.weight": [256, 741],
        "network.0.bias": [256],
        "network.2.weight": [128, 256],
        "network.2.bias": [128],
        "network.4.weight": [1, 128],
        "network.4.bias": [1],
    },
}

# The checkpoint keys written by ``q1_soccer.train.save`` at the same baseline
# commit, before any pole profile or source manifest existed.
PRE_REFACTOR_SAVE_KEYS = {
    "format",
    "iteration",
    "config",
    "vendor_commit",
    "actor",
    "model",
    "critic",
    "actor_optimizer",
    "model_optimizer",
    "critic_optimizer",
    "baseline",
    "torch_rng",
    "cuda_rng",
    "replay_rng",
    "validation_rng",
    "online_seed",
    "online_archives",
    "training_revision",
    "training_data_policy",
    "accepted_policy_updates",
    "consecutive_rejections",
}

POLE_ONLY_SETTINGS = {
    "output_root",
    "profile",
    "source_manifest",
    "source_root",
    "source_revision",
    "warmup_checkpoint",
    "warmup_checkpoint_id",
    "course_conditioned",
    "goal_representation",
    "modeled_action_delay",
    "pole_clearance_weight",
}


def _soccer_networks():
    robot = SoccerRobot()
    # Same construction as ``q1_soccer.train.build``: the teacher configuration
    # comes from the robot's native Q1.yaml and is not pole-conditioned.
    actor = VisionPolicy(robot.cfg)
    model = SoccerWorldModel(robot)
    critic = SoccerCritic(robot)
    return actor, model, critic


def _shapes(module) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in module.state_dict().items()}


def test_soccer_profile_pins_the_pre_refactor_architecture():
    profile = resolve_profile()
    actor, model, critic = _soccer_networks()

    assert profile.name == "soccer"
    assert profile.model_contract == "soccer-vision-79x50-v1"
    assert profile.checkpoint_format == "primebot_supermimic_soccer_v1"
    assert _shapes(actor) == SOCCER_STATE_DICT_SHAPES["actor"]
    assert _shapes(model) == SOCCER_STATE_DICT_SHAPES["model"]
    assert _shapes(critic) == SOCCER_STATE_DICT_SHAPES["critic"]
    assert actor.encoder[0].in_features == 79 * 50
    assert model.dynamics[0].in_features == 109 + 22
    assert model.dynamics[-1].out_features == 87
    # 109 state + 10 x 57 goal features + 62 native context values.
    assert critic.network[0].in_features == 109 + 10 * 57 + 62 == 741
    assert not hasattr(actor, "speed_encoder")
    assert not hasattr(actor, "route_encoder")


def test_pre_refactor_soccer_checkpoint_round_trips_without_pole_settings(tmp_path):
    from q1_soccer.train import TrainConfig, save

    actor, model, critic = _soccer_networks()
    algorithm = SimpleNamespace(
        actor=actor,
        model=model,
        critic=critic,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=1e-3),
        model_optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.SGD(critic.parameters(), lr=1e-3),
    )
    replay = SimpleNamespace(
        generator=torch.Generator().manual_seed(1),
        online_archives=["runs/example/online-0"],
    )
    validation = SimpleNamespace(generator=torch.Generator().manual_seed(2))
    cfg = TrainConfig(device="cpu")
    checkpoint_path = tmp_path / "pre-refactor-500.pt"

    save(
        checkpoint_path,
        algorithm,
        cfg,
        iteration=500,
        baseline={"success": 0.5},
        replay=replay,
        validation=validation,
        online_seed=3000000,
        trust=SimpleNamespace(accepted_updates=3, consecutive_rejections=1),
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(saved) == PRE_REFACTOR_SAVE_KEYS
    assert saved["format"] == resolve_profile("soccer").checkpoint_format
    assert saved["training_revision"] == "real_critic_trust_v2"
    assert saved["training_data_policy"] == "strict_success_only_v1"
    assert saved["accepted_policy_updates"] == 3
    assert saved["consecutive_rejections"] == 1
    assert set(saved["config"]) & POLE_ONLY_SETTINGS == set()
    assert "output" in saved["config"] and "checkpoint" in saved["config"]

    reloaded_actor, reloaded_model, reloaded_critic = _soccer_networks()
    reloaded_actor.load_state_dict(saved["actor"], strict=True)
    reloaded_model.load_state_dict(saved["model"], strict=True)
    reloaded_critic.load_state_dict(saved["critic"], strict=True)

    for name in SOCCER_STATE_DICT_SHAPES:
        original = algorithm.actor if name == "actor" else getattr(algorithm, name)
        reloaded = {
            "actor": reloaded_actor,
            "model": reloaded_model,
            "critic": reloaded_critic,
        }[name]
        for key, value in original.state_dict().items():
            assert torch.equal(reloaded.state_dict()[key], value)
    assert not any("pole" in key or "course" in key for key in saved["actor"])
    assert not any("pole" in key or "course" in key for key in saved["model"])
    assert not any("pole" in key or "course" in key for key in saved["critic"])


def test_soccer_profile_does_not_resolve_to_a_pole_run_directory():
    soccer = resolve_profile()
    pole = resolve_profile("pole-weaving")

    assert soccer.output_root != pole.output_root
    assert soccer.checkpoint_format != pole.checkpoint_format
    assert soccer.config_path != pole.config_path
    assert Path(soccer.output_root).name == "auto"
    assert Path(pole.output_root).name == "q1-pole-weaving"
