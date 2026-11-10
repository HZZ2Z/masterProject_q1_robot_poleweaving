"""Resolve and enforce the registered source task 7.7 training recipe."""

from __future__ import annotations

import json
import math

import pytest
import torch

from pole_task import COURSE_ROOT, ROOT
from pole_task.models import build_source_models
from pole_task.source import source_robot_xml
from pole_task.source_profile import SourcePoleRobot
from pole_task.source_recipe import (
    SOURCE_ACTOR_LATENT,
    freeze_pole_input_columns,
    pole_input_columns,
    resolve_source_recipe,
)

CONFIG = COURSE_ROOT / "pole_task/config.json"
MANIFEST = COURSE_ROOT / "pole_task/source_manifest.json"


def _recipe():
    return resolve_source_recipe(CONFIG)


def test_resolved_recipe_matches_the_registered_stage_transition():
    recipe = _recipe()
    assert recipe.format == "supermimic_source_task_7_7_recipe_v1"
    assert recipe.source_revision == "e023c3147e5c20af17f329e0b467538f627a27cb"
    assert recipe.warmup_checkpoint_id == "warmup_checkpoint"
    assert (
        recipe.warmup_checkpoint_sha256
        == "b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5"
    )
    assert recipe.final_checkpoint_id == "passed_final_checkpoint_reference"
    assert (
        recipe.final_checkpoint_sha256
        == "c42237dff52a5664d83af37fa7afed00670d09955f562855161dc59f7b73f8d6"
    )

    warmup, actor = recipe.stages
    assert (warmup.name, warmup.updates, warmup.start_update, warmup.end_update) == (
        "warmup",
        2000,
        1,
        2000,
    )
    assert not warmup.actor_updates_enabled
    assert not warmup.randomize
    assert not warmup.freeze_pole_input_columns
    assert (actor.name, actor.updates, actor.start_update, actor.end_update) == (
        "actor",
        500,
        2001,
        2500,
    )
    assert actor.actor_updates_enabled
    assert actor.randomize
    assert actor.freeze_pole_input_columns
    assert recipe.total_updates == 2500
    assert recipe.actor_updates == 500
    assert recipe.stage_for_update(2000) is warmup
    assert recipe.stage_for_update(2001) is actor
    with pytest.raises(ValueError, match="outside the source recipe"):
        recipe.stage_for_update(2501)


def test_resolved_optimizers_learning_rates_and_horizon_match_source_configs():
    recipe = _recipe()
    assert recipe.imagination_horizon == 32
    assert recipe.model_optimizer.learning_rate == 1e-4
    assert recipe.critic_optimizer.learning_rate == 1e-4
    assert recipe.actor_optimizer.learning_rate == 1e-5
    assert recipe.model_optimizer.betas == (0.9, 0.999)
    assert recipe.critic_optimizer.betas == (0.9, 0.999)
    assert recipe.actor_optimizer.betas == (0.9, 0.95)
    assert all(
        optimizer.name == "adam"
        and optimizer.eps == 1e-8
        and optimizer.weight_decay == 0.0
        for optimizer in (
            recipe.model_optimizer,
            recipe.critic_optimizer,
            recipe.actor_optimizer,
        )
    )
    assert (recipe.model_batches, recipe.critic_epochs, recipe.critic_minibatches) == (
        10,
        5,
        4,
    )
    assert recipe.action_scale == 0.5
    assert recipe.clean_policy_actions
    assert recipe.modeled_action_delay

    assert recipe.learning_rate_factor(0) == pytest.approx(1.0)
    assert recipe.learning_rate_factor(250_000) == pytest.approx(0.03)
    rates = recipe.optimizer_rates(0)
    assert rates == pytest.approx({"model": 1e-4, "actor": 1e-5, "critic": 1e-4})
    assert recipe.learning_rate_factor(10) == pytest.approx(
        0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * 10 / 250_000))
    )

    document = json.loads(json.dumps(recipe.to_dict()))
    assert document["stages"][1]["freeze_pole_input_columns"] is True
    assert document["optimizers"]["actor"]["betas"] == [0.9, 0.95]
    assert document["learning_rate_schedule"] == {
        "kind": "cosine",
        "period_updates": 250_000,
        "floor": 0.03,
    }


def test_frozen_pole_input_columns_match_the_actor_decoder_layout():
    robot = SourcePoleRobot(source_robot_xml(MANIFEST))
    actor, _, _ = build_source_models(robot)
    first = actor.decoder[0]
    assert first.in_features == SOURCE_ACTOR_LATENT + 10 * robot.obs_dim
    columns = pole_input_columns()
    assert len(columns) == 60
    assert len(set(columns)) == 60
    assert columns[:6] == tuple(range(104, 110))
    assert columns[-6:] == tuple(range(806, 812))
    assert all(index < first.in_features for index in columns)

    keys = set(actor.state_dict())
    installed = freeze_pole_input_columns(actor)
    assert installed == columns
    assert set(actor.state_dict()) == keys

    first.weight.grad = None
    loss = first.weight[:, 0].sum() + first.weight[:, list(columns)].sum()
    loss.backward()
    gradient = first.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[:, 0]) > 0
    assert torch.count_nonzero(gradient[:, list(columns)]) == 0

    # Installation is idempotent for the same layout.
    freeze_pole_input_columns(actor)


def _mutated_config(tmp_path, mutate):
    config = json.loads(CONFIG.read_text())
    mutate(config["source_training"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda training: training["stages"]["actor"].update({"updates": 501}),
            "Actor update budget",
        ),
        (
            lambda training: training.update({"imagination_horizon": 16}),
            "Imagination horizon",
        ),
        (
            lambda training: training.update({"actor_learning_rate": 1e-4}),
            "[Aa]ctor learning rate",
        ),
        (
            lambda training: training["stages"]["actor"].update(
                {"freeze_pole_input_columns": False}
            ),
            "Actor-stage pole-input freezing",
        ),
        (
            lambda training: training["actor_optimizer"].update(
                {"betas": [0.9, 0.999]}
            ),
            "Actor optimizer betas",
        ),
    ],
)
def test_resolved_recipe_rejects_drift_from_the_source_contract(
    tmp_path, mutate, message
):
    path = _mutated_config(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        resolve_source_recipe(path, project_root=ROOT)
