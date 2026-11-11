# Q1 Robot Soccer

English | [简体中文](README.zh-CN.md)

**Simulation and learning for Q1 humanoid robot control, soccer kicking, and locomotion.**

Q1 Robot Soccer is a research and development project built around the Qiyuan Q1 humanoid robot. It brings together simulation environments, training configurations, policy playback, and evaluation tools for robot control, imitation learning, and reinforcement learning. Start from an existing task to reproduce an experiment, or modify observations, rewards, policy networks, and training workflows to explore new behaviors.

![Soccer kicking policy demonstration](/docs/assets/demos/football-kicking.gif)
![S-shaped pole weaving policy demonstration](/docs/assets/demos/pole-weaving.gif)

*Policy demonstrations for kicking and S-shaped pole weaving. See the course pages below for task details and additional demos.*

[Features](#features) · [Courses](#courses) · [Quick Start](#quick-start) · [Requirements](#requirements) · [Project Structure](#project-structure) · [Development](#development)

## Features

- **Multiple robot tasks:** Soccer kicking and continuous pole weaving provide different entry points for locomotion and policy research.
- **Multiple learning workflows:** Expert trajectory collection, imitation learning, world-model-based policy training, and staged reinforcement learning.
- **Training, playback, and evaluation:** Each task provides its own configurations and commands for running pipelines, loading policies, and recording experiments.
- **Shared simulation and robot resources:** Tasks reuse MuJoCo environments, robot models, and common training code. Editable installation supports direct source development.

## Courses

| Course | Goal | Methods and practice |
| --- | --- | --- |
| [Soccer Kicking](courses/football-kicking/README.md) | Approach and kick a ball using observations | Expert trajectory collection, imitation learning, and online policy updates |
| [S-shaped Pole Weaving](courses/pole-weaving/README.md) | Navigate continuously around a three-pole course | SuperMimic world model, observation design, and reward design |

Each course README is available in English and Chinese. Linked supporting guides may currently be in Chinese. Each course has its own configurations, training workflow, and resource requirements. Course scope describes the training task design; actual policy capabilities should be assessed using checkpoints and evaluation results.

## Quick Start

### 1. Set up the base environment

Prepare Linux and Conda or Mamba, then run these commands from the repository root:

```bash
# Initialize third-party submodules
git submodule update --init --recursive

# Create and activate the Python 3.11 environment
conda env create -f environment.yml
conda activate q1-kick-4v4

# Install the project in editable mode
python -m pip install --no-deps -e .
```

See the [environment setup guide](docs/setup.md) for dependency downloads, CUDA requirements, and installation checks. The guide is currently in Chinese.

### 2. Choose a course and prepare its resources

- **Soccer kicking and imitation learning:** Open [Soccer Kicking](courses/football-kicking/README.md). Prepare the expert checkpoint before collecting short trajectories or using the full dataset.
- **World models and locomotion:** Open [S-shaped Pole Weaving](courses/pole-weaving/README.md). Prepare the task-specific warmup checkpoint as described there.

### 3. Follow the selected course

Use the selected course page for its first-run instructions, resource preparation, training, playback, and evaluation commands. Some linked supporting guides are currently in Chinese.

## Requirements

The shared base environment uses **Linux, Python 3.11, MuJoCo, and PyTorch**. The current environment installs a CUDA 12.8 build of PyTorch. GPU workflows require an NVIDIA GPU and a compatible driver.

| Course | CPU and GPU support | Additional models or data |
| --- | --- | --- |
| Soccer Kicking | CPU smoke training; CUDA for full training | Expert checkpoint for smoke workflows; full successful-trajectory dataset for full training |
| S-shaped Pole Weaving | Training defaults to an NVIDIA GPU; evaluation and demos default to CPU | Task 7.7 warmup checkpoint for training; existing course checkpoints can be evaluated or played back |

Task-specific weights and datasets may not be distributed in Git. Check each course page before running its workflows. See the [pole-weaving reproduction guide](docs/pole-weaving.md) for asset registration and reproduction details (currently in Chinese).

## Project Structure

```text
Q1-Robot-Soccer/
├── courses/                 # Task configurations, implementations, and entry points
│   ├── football-kicking/    # Kicking configuration and task guide
│   ├── pole-weaving/        # S-shaped pole-weaving environment, training, and evaluation
│   └── soccer-4v4/          # 4v4 simulation, training, and match logic
├── shared/                  # Code and resources shared across tasks
│   ├── q1-mujoco/           # MuJoCo environments, robot models, and tools
│   └── supermimic/          # Shared training code, policies, and expert tools
├── 3rdparty/                # Third-party dependencies
├── docs/                    # Setup, reproduction, maintenance, and demo resources
├── environment.yml          # Conda environment definition
└── requirements.txt         # Python dependencies
```

The root `setup.py` maps Python packages to source under `courses/` and `shared/`. After editable installation, source changes are available to subsequent runs. See the [maintenance inventory](docs/maintenance/course-file-inventory.md) for file ownership and standard commands (currently in Chinese).

## Development

- **Modify tasks and rewards:** Update observations, rewards, resets, and termination conditions in the relevant task implementation. Compare behavior with consistent evaluation settings.
- **Improve policies and training:** Start from course configurations and training code. Record checkpoints and experiment settings to support reproducibility.
- **Extend simulation and robot resources:** Find shared environments and models in `shared/q1-mujoco/`, and maintain task-specific logic in each course directory.

Soccer kicking uses the `q1_soccer` training package under `shared/supermimic/`; task-specific implementations live in their course directories. See the [maintenance inventory](docs/maintenance/course-file-inventory.md) for the complete module mapping. Training data, checkpoints, and logs are stored locally under `logs/` or task-specific directories and are not tracked by Git.

## Related Projects

The project uses simulation and training components including MuJoCo, PyTorch, and Warp, and includes related code from [SuperMimic](3rdparty/SuperMimic/README.md). Refer to each component's directory for its origin, usage, and license.

## License

See [LICENSE](LICENSE) for the project license. Third-party components retain their respective licenses.
