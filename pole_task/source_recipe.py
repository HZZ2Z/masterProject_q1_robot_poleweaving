"""Resolve the registered source task 7.7 warmup-to-actor training recipe.

The source task trained a frozen-actor course warmup for 2000 updates and then
enabled the actor for the next 500 updates while training the same world model
and critic.  This module encodes that transition explicitly and checks every
resolved value against the registered source configs and migration manifest so
the project cannot silently train with a different horizon, budget, learning
rate, optimizer, or pole-input trust region.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib

import torch
from torch import Tensor, nn

from .source import verify_source_manifest
from .source_profile import (
    SOURCE_HISTORY_FRAMES,
    SOURCE_OBS_DIM,
    SOURCE_POLE_FEATURES,
)

SOURCE_ACTOR_LATENT = 32
SOURCE_LEARNING_RATE_PERIOD = 250_000
SOURCE_LEARNING_RATE_FLOOR = 0.03


@dataclass(frozen=True)
class OptimizerSettings:
    """Resolved Adam settings for one source optimizer."""

    name: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class SourceStage:
    """A contiguous, one-based range of source training updates."""

    name: str
    updates: int
    start_update: int
    end_update: int
    actor_updates_enabled: bool
    randomize: bool
    freeze_pole_input_columns: bool


@dataclass(frozen=True)
class SourceRecipe:
    """The resolved task 7.7 recipe used by the pole profile."""

    format: str
    source_revision: str
    warmup_checkpoint_id: str
    warmup_checkpoint_sha256: str
    final_checkpoint_id: str
    final_checkpoint_sha256: str
    stages: tuple[SourceStage, ...]
    model_optimizer: OptimizerSettings
    critic_optimizer: OptimizerSettings
    actor_optimizer: OptimizerSettings
    imagination_horizon: int
    model_batch_size: int
    policy_batch_size: int
    model_batches: int
    critic_epochs: int
    critic_minibatches: int
    action_scale: float
    clean_policy_actions: bool
    modeled_action_delay: bool
    learning_rate_period_updates: int
    learning_rate_floor: float

    @property
    def total_updates(self) -> int:
        return sum(stage.updates for stage in self.stages)

    @property
    def actor_updates(self) -> int:
        return sum(stage.updates for stage in self.stages if stage.actor_updates_enabled)

    @property
    def freeze_pole_input_columns(self) -> bool:
        return any(stage.freeze_pole_input_columns for stage in self.stages)

    def stage_for_update(self, update: int) -> SourceStage:
        """Return the one-based stage that owns a global update index."""
        for stage in self.stages:
            if stage.start_update <= update <= stage.end_update:
                return stage
        raise ValueError(
            f"Update {update} is outside the source recipe's 1..{self.total_updates} budget"
        )

    def learning_rate_factor(self, iteration: int) -> float:
        """Source cosine factor applied to every optimizer before an update.

        ``iteration`` is the zero-based source loop index, matching
        ``configs/q1-pole-weaving-actor-v10-block7.toml``.
        """
        if iteration < 0:
            raise ValueError("Learning-rate iteration must be nonnegative")
        progress = min(iteration, self.learning_rate_period_updates) / self.learning_rate_period_updates
        return self.learning_rate_floor + (1.0 - self.learning_rate_floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def optimizer_rates(self, iteration: int) -> dict[str, float]:
        """Learning rates scheduled for the model, actor, and critic."""
        factor = self.learning_rate_factor(iteration)
        return {
            "model": self.model_optimizer.learning_rate * factor,
            "actor": self.actor_optimizer.learning_rate * factor,
            "critic": self.critic_optimizer.learning_rate * factor,
        }

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "source_revision": self.source_revision,
            "warmup_checkpoint_id": self.warmup_checkpoint_id,
            "warmup_checkpoint_sha256": self.warmup_checkpoint_sha256,
            "final_checkpoint_id": self.final_checkpoint_id,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "stages": [
                {
                    "name": stage.name,
                    "updates": stage.updates,
                    "start_update": stage.start_update,
                    "end_update": stage.end_update,
                    "actor_updates_enabled": stage.actor_updates_enabled,
                    "randomize": stage.randomize,
                    "freeze_pole_input_columns": stage.freeze_pole_input_columns,
                }
                for stage in self.stages
            ],
            "optimizers": {
                "model": _optimizer_dict(self.model_optimizer),
                "actor": _optimizer_dict(self.actor_optimizer),
                "critic": _optimizer_dict(self.critic_optimizer),
            },
            "imagination_horizon": self.imagination_horizon,
            "model_batch_size": self.model_batch_size,
            "policy_batch_size": self.policy_batch_size,
            "model_batches": self.model_batches,
            "critic_epochs": self.critic_epochs,
            "critic_minibatches": self.critic_minibatches,
            "action_scale": self.action_scale,
            "clean_policy_actions": self.clean_policy_actions,
            "modeled_action_delay": self.modeled_action_delay,
            "learning_rate_schedule": {
                "kind": "cosine",
                "period_updates": self.learning_rate_period_updates,
                "floor": self.learning_rate_floor,
            },
        }


def _optimizer_dict(settings: OptimizerSettings) -> dict:
    return {
        "name": settings.name,
        "learning_rate": settings.learning_rate,
        "betas": list(settings.betas),
        "eps": settings.eps,
        "weight_decay": settings.weight_decay,
    }


def _expect(actual, expected, label: str) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} must be {expected}, got {actual}")
    elif actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _stage(name: str, values: dict, start_update: int) -> SourceStage:
    updates = int(values["updates"])
    if updates < 0:
        raise ValueError(f"{name} stage updates must be nonnegative")
    return SourceStage(
        name=name,
        updates=updates,
        start_update=start_update,
        end_update=start_update + updates - 1,
        actor_updates_enabled=bool(values["actor_updates_enabled"]),
        randomize=bool(values["randomize"]),
        freeze_pole_input_columns=bool(values["freeze_pole_input_columns"]),
    )


def _optimizer(
    name: str,
    values: dict,
    learning_rate: float,
) -> OptimizerSettings:
    if not isinstance(values, dict):
        raise ValueError(f"{name} optimizer settings must be an object")
    betas = tuple(float(value) for value in values["betas"])
    if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
        raise ValueError(f"{name} optimizer betas must contain two values in [0, 1)")
    if values["name"] != "adam":
        raise ValueError(f"{name} optimizer must be Adam, got {values['name']!r}")
    if float(values["weight_decay"]) != 0.0:
        raise ValueError(f"{name} optimizer must keep zero weight decay")
    return OptimizerSettings(
        name="adam",
        learning_rate=learning_rate,
        betas=betas,
        eps=float(values["eps"]),
        weight_decay=0.0,
    )


def _source_train_config(artifact_paths: dict[str, str], artifact_id: str) -> dict:
    path = artifact_paths.get(artifact_id)
    if path is None:
        raise ValueError(f"Source manifest does not register {artifact_id!r}")
    document = tomllib.loads(Path(path).read_text())
    train = document.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"Source config {artifact_id!r} is missing its [train] table")
    return train


def resolve_source_recipe(
    config_path: Path,
    *,
    project_root: Path | None = None,
) -> SourceRecipe:
    """Resolve and cross-check the task 7.7 recipe from project config + manifest."""
    config_path = Path(config_path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else config_path.parents[1]
    )
    config = json.loads(config_path.read_text())
    settings = config.get("source_training")
    if not isinstance(settings, dict):
        raise ValueError("Pole config must define a source_training recipe")
    source_manifest_path = (root / config["source_manifest"]).resolve()
    report = verify_source_manifest(source_manifest_path)
    manifest = json.loads(source_manifest_path.read_text())
    training = manifest["training_contract"]

    _expect(settings["source_revision"], manifest["source_revision"], "Source revision")
    warmup_train = _source_train_config(report["artifact_paths"], "warmup_config")
    actor_train = _source_train_config(report["artifact_paths"], "actor_config")

    stage_settings = settings.get("stages")
    if not isinstance(stage_settings, dict) or tuple(stage_settings) != ("warmup", "actor"):
        raise ValueError("Source recipe stages must be the ordered pair (warmup, actor)")
    warmup = _stage("warmup", stage_settings["warmup"], start_update=1)
    actor = _stage("actor", stage_settings["actor"], start_update=warmup.updates + 1)
    stages = (warmup, actor)

    _expect(warmup.updates, int(training["warmup_updates"]), "Warmup update budget")
    _expect(warmup.updates, int(warmup_train["iterations"]), "Warmup source iterations")
    _expect(
        warmup.actor_updates_enabled,
        int(warmup_train["actor_warmup_updates"]) < warmup.updates,
        "Warmup actor updates",
    )
    _expect(warmup.randomize, bool(warmup_train["randomize"]), "Warmup randomization")
    _expect(
        warmup.freeze_pole_input_columns,
        bool(warmup_train.get("freeze_pole_input_columns", False)),
        "Warmup pole-input freezing",
    )

    _expect(actor.updates, int(training["actor_updates"]), "Actor update budget")
    _expect(
        actor.updates,
        int(actor_train["iterations"]) - warmup.updates,
        "Actor source iterations after the warmup checkpoint",
    )
    _expect(
        actor.actor_updates_enabled,
        int(actor_train["actor_warmup_updates"]) == 0,
        "Actor-stage actor updates",
    )
    _expect(actor.randomize, bool(actor_train["randomize"]), "Actor-stage randomization")
    _expect(
        actor.freeze_pole_input_columns,
        bool(actor_train.get("freeze_pole_input_columns", False)),
        "Actor-stage pole-input freezing",
    )
    _expect(
        actor.freeze_pole_input_columns,
        bool(training["freeze_pole_input_columns"]),
        "Manifest pole-input freezing",
    )

    model_learning_rate = float(settings["model_learning_rate"])
    critic_learning_rate = float(settings["critic_learning_rate"])
    actor_learning_rate = float(settings["actor_learning_rate"])
    _expect(model_learning_rate, float(warmup_train["learning_rate"]), "Warmup model learning rate")
    _expect(model_learning_rate, float(actor_train["learning_rate"]), "Actor-stage model learning rate")
    _expect(critic_learning_rate, float(actor_train["learning_rate"]), "Critic learning rate")
    _expect(actor_learning_rate, float(warmup_train["actor_learning_rate"]), "Warmup actor learning rate")
    _expect(actor_learning_rate, float(actor_train["actor_learning_rate"]), "Actor learning rate")
    _expect(actor_learning_rate, float(training["actor_learning_rate"]), "Manifest actor learning rate")

    imagination_horizon = int(settings["imagination_horizon"])
    _expect(imagination_horizon, int(training["imagination_horizon"]), "Imagination horizon")
    _expect(
        imagination_horizon,
        int(warmup_train.get("imagination_horizon", 32)),
        "Warmup source imagination horizon",
    )
    _expect(
        imagination_horizon,
        int(actor_train.get("imagination_horizon", 32)),
        "Actor source imagination horizon",
    )

    model_batch_size = int(settings["model_batch_size"])
    policy_batch_size = int(settings["policy_batch_size"])
    model_batches = int(settings["model_batches"])
    critic_epochs = int(settings["critic_epochs"])
    critic_minibatches = int(settings["critic_minibatches"])
    for label, value, source_value in (
        ("Model batch size", model_batch_size, actor_train["model_batch_size"]),
        ("Policy batch size", policy_batch_size, actor_train["policy_batch_size"]),
        ("Model batches", model_batches, actor_train["model_batches"]),
        ("Critic epochs", critic_epochs, actor_train["critic_epochs"]),
        ("Critic minibatches", critic_minibatches, actor_train["critic_minibatches"]),
    ):
        _expect(value, int(source_value), label)
    if min(model_batch_size, policy_batch_size, model_batches, critic_epochs, critic_minibatches) <= 0:
        raise ValueError("Source batch and critic update sizes must be positive")

    action_scale = float(settings["action_scale"])
    _expect(action_scale, float(warmup_train["action_scale"]), "Warmup action scale")
    _expect(action_scale, float(actor_train["action_scale"]), "Actor action scale")
    _expect(bool(settings["clean_policy_actions"]), bool(actor_train["clean_policy_actions"]), "Clean policy actions")
    _expect(bool(settings["modeled_action_delay"]), bool(actor_train["modeled_action_delay"]), "Modeled action delay")

    schedule = settings.get("learning_rate_schedule")
    if not isinstance(schedule, dict) or schedule.get("kind") != "cosine":
        raise ValueError("Source recipe requires the registered cosine learning-rate schedule")
    period = int(schedule["period_updates"])
    floor = float(schedule["floor"])
    if period <= 0 or not 0.0 <= floor <= 1.0:
        raise ValueError("Cosine learning-rate period/floor are invalid")
    _expect(period, SOURCE_LEARNING_RATE_PERIOD, "Learning-rate period")
    _expect(floor, SOURCE_LEARNING_RATE_FLOOR, "Learning-rate floor")

    model_optimizer = _optimizer(
        "Model", settings["model_optimizer"], model_learning_rate
    )
    critic_optimizer = _optimizer(
        "Critic", settings["critic_optimizer"], critic_learning_rate
    )
    actor_optimizer = _optimizer(
        "Actor", settings["actor_optimizer"], actor_learning_rate
    )
    _expect(model_optimizer.betas, (0.9, 0.999), "Model optimizer betas")
    _expect(critic_optimizer.betas, (0.9, 0.999), "Critic optimizer betas")
    _expect(actor_optimizer.betas, (0.9, 0.95), "Actor optimizer betas")

    return SourceRecipe(
        format="supermimic_source_task_7_7_recipe_v1",
        source_revision=report["source_revision"],
        warmup_checkpoint_id=report["warmup_checkpoint_id"],
        warmup_checkpoint_sha256=next(
            artifact["sha256"]
            for artifact in manifest["artifacts"]
            if artifact["id"] == report["warmup_checkpoint_id"]
        ),
        final_checkpoint_id=report["final_checkpoint_reference_id"],
        final_checkpoint_sha256=next(
            artifact["sha256"]
            for artifact in manifest["artifacts"]
            if artifact["id"] == report["final_checkpoint_reference_id"]
        ),
        stages=stages,
        model_optimizer=model_optimizer,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        imagination_horizon=imagination_horizon,
        model_batch_size=model_batch_size,
        policy_batch_size=policy_batch_size,
        model_batches=model_batches,
        critic_epochs=critic_epochs,
        critic_minibatches=critic_minibatches,
        action_scale=action_scale,
        clean_policy_actions=bool(settings["clean_policy_actions"]),
        modeled_action_delay=bool(settings["modeled_action_delay"]),
        learning_rate_period_updates=period,
        learning_rate_floor=floor,
    )


def pole_input_columns(
    *,
    obs_dim: int = SOURCE_OBS_DIM,
    history: int = SOURCE_HISTORY_FRAMES,
    latent_dim: int = SOURCE_ACTOR_LATENT,
) -> tuple[int, ...]:
    """Decoder-input columns holding the six pole coordinates of each frame."""
    if obs_dim < SOURCE_POLE_FEATURES or history < 1 or latent_dim < 1:
        raise ValueError("Actor decoder dimensions cannot hold pole inputs")
    columns: list[int] = []
    for frame in range(history):
        start = latent_dim + (frame + 1) * obs_dim - SOURCE_POLE_FEATURES
        columns.extend(range(start, start + SOURCE_POLE_FEATURES))
    return tuple(columns)


def freeze_pole_input_columns(
    actor: nn.Module,
    *,
    obs_dim: int = SOURCE_OBS_DIM,
    history: int = SOURCE_HISTORY_FRAMES,
) -> tuple[int, ...]:
    """Zero the actor decoder's gradient for every pole-coordinate input.

    This mirrors source task 7.7's parameter trust region without adding a
    buffer, so the actor state dict remains identical to the registered source
    checkpoint contract.
    """
    decoder = getattr(actor, "decoder", None)
    if decoder is None or len(decoder) == 0 or not isinstance(decoder[0], nn.Linear):
        raise TypeError("Source actor decoder must begin with a linear layer")
    first = decoder[0]
    latent = first.in_features - history * obs_dim
    if latent != SOURCE_ACTOR_LATENT:
        raise ValueError(
            f"Source actor decoder expects {SOURCE_ACTOR_LATENT + history * obs_dim} "
            f"inputs, got {first.in_features}"
        )
    columns = pole_input_columns(obs_dim=obs_dim, history=history, latent_dim=latent)
    mask = torch.ones_like(first.weight)
    mask[:, list(columns)] = 0.0
    existing = getattr(actor, "_pole_input_gradient_mask", None)
    if existing is not None:
        if existing.shape != mask.shape or not torch.equal(existing, mask):
            raise ValueError("Actor already carries a different pole-input gradient mask")
        return columns
    actor._pole_input_gradient_mask = mask
    first.weight.register_hook(lambda gradient: gradient * mask.to(gradient.device))
    return columns


__all__ = [
    "OptimizerSettings",
    "SOURCE_ACTOR_LATENT",
    "SOURCE_LEARNING_RATE_FLOOR",
    "SOURCE_LEARNING_RATE_PERIOD",
    "SourceRecipe",
    "SourceStage",
    "freeze_pole_input_columns",
    "pole_input_columns",
    "resolve_source_recipe",
]
