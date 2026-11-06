import json
from types import ModuleType

import pytest

from q1_soccer.profiles import resolve_profile


def test_default_profile_keeps_existing_soccer_config_and_contract():
    profile = resolve_profile()

    assert profile.name == "soccer"
    assert profile.config_path.name == "soccer.json"
    assert profile.output_root.name == "auto"
    assert profile.model_contract == "soccer-vision-79x50-v1"
    assert profile.checkpoint_format == "primebot_supermimic_soccer_v1"


def test_pole_profile_selects_source_contract_and_separate_output():
    profile = resolve_profile("pole-weaving")

    assert profile.config_path.name == "config.json"
    assert profile.output_root.name == "q1-pole-weaving"
    assert profile.model_contract == "source-q1-course-v1"
    assert profile.checkpoint_format == "supermimic_pole_weaving_profile_v1"


def test_shared_training_entrypoint_dispatches_pole_profile(monkeypatch):
    from q1_supermimic import train

    captured = {}

    def pole_main(argv):
        captured["argv"] = argv
        return 7

    pole_module = ModuleType("pole_task.training")
    pole_module.main = pole_main
    monkeypatch.setitem(__import__("sys").modules, "pole_task.training", pole_module)

    result = train.main(["--profile", "pole-weaving", "--check"])

    assert result == 7
    assert captured["argv"] == [
        "--config",
        str(resolve_profile("pole-weaving").config_path),
        "--preflight-only",
    ]


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("soccer", {"output": "logs/q1-pole-weaving"}),
        ("pole-weaving", {"output_root": "logs/auto"}),
    ],
)
def test_profile_rejects_output_path_owned_by_other_profile(tmp_path, name, config):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="overlaps the .* profile output"):
        resolve_profile(name, path, project_root=tmp_path)
