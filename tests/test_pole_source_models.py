import json
from types import SimpleNamespace

import torch

from pole_task import COURSE_ROOT, ROOT
from pole_task.source import verify_source_manifest


def test_source_profile_models_match_registered_checkpoint_tensor_contract():
    manifest_path = COURSE_ROOT / "pole_task/source_manifest.json"
    source = verify_source_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    robot = SimpleNamespace(
        state_dim=79,
        physical_dim=57,
        obs_dim=manifest["robot_contract"]["per_frame_observation_dim"],
        context_dim=149,
        modeled_action_delay=True,
        context_layout=SimpleNamespace(delay_s=slice(0, 22)),
        reward_scaling=SimpleNamespace(termination_height=0.25, termination_tilt=0.8),
        default_q=torch.zeros(22),
    )

    from pole_task.models import build_source_models

    actor, world_model, critic = build_source_models(robot)

    for name, module in (("actor", actor), ("model", world_model), ("critic", critic)):
        actual = {key: list(value.shape) for key, value in module.state_dict().items()}
        assert actual == source["tensor_shapes"][name]
    assert actor(torch.zeros(2, 10, 78), torch.zeros(2, 570)).shape == (2, 22)
