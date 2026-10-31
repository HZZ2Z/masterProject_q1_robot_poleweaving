"""Open MuJoCo playback, finding a suitable local Python environment if needed."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def ensure_environment():
    dependencies = ("torch", "mujoco", "yaml")
    if all(importlib.util.find_spec(name) is not None for name in dependencies):
        return
    root = Path(__file__).resolve().parent
    prefixes = [Path(sys.prefix)]
    for name in ("CONDA_PREFIX", "CONDA_EXE"):
        if os.environ.get(name):
            path = Path(os.environ[name])
            prefixes.append(path.parents[1] if name == "CONDA_EXE" else path)
    prefixes.extend([Path.home() / "miniconda3", Path.home() / "anaconda3"])
    candidates = [root / ".venv/bin/python"]
    for prefix in prefixes:
        candidates.extend(
            prefix / "envs" / name / "bin/python"
            for name in ("q1-supermimic", "q1-kick")
        )
        if prefix.parent.name == "envs":
            candidates.extend(
                prefix.parent / name / "bin/python"
                for name in ("q1-supermimic", "q1-kick")
            )
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if (
            candidate in checked
            or candidate == Path(sys.executable).resolve()
            or not candidate.is_file()
        ):
            continue
        checked.add(candidate)
        try:
            probe = subprocess.run(
                [str(candidate), "-c", "import torch, mujoco, yaml"],
                env=environment,
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            print(f"Playback Python: {candidate}", flush=True)
            os.execve(
                str(candidate),
                [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
                environment,
            )
    raise SystemExit(
        "当前 Python 缺少 PyTorch/MuJoCo，且未找到可用环境。请先 conda activate q1-kick，或按 README 安装 q1-supermimic 环境。"
    )


def main():
    ensure_environment()
    from q1_soccer.play import main as playback_main

    playback_main()


if __name__ == "__main__":
    main()
