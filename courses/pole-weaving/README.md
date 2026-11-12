# S-shaped Pole Weaving

English | [简体中文](README.zh-CN.md)

This course provides a Q1 baseline for weaving through a three-pole S-shaped course using the SuperMimic world-model framework. It covers the task environment, observation design, and reward design, and supports baseline reproduction and policy improvement.

![S-shaped pole weaving policy demonstration](../../docs/assets/demos/pole-weaving.gif)

## Before You Start

Set up the environment using the [shared environment setup guide](../../docs/setup.md) (currently in Chinese). Training defaults to `cuda:0` and requires an NVIDIA GPU with at least 4 GB of free memory. Evaluation and demonstrations default to CPU. Training requires the task 7.7 warmup checkpoint; download the [pole warmup checkpoint](https://drive.google.com/file/d/1PmM1NoD09gKIwzqlj3-rCeQRdffVutjJ/view?usp=sharing) and register it at the exact path described in the [asset and reproduction guide](../../docs/pole-weaving.md) (currently in Chinese). It cannot be replaced with the soccer expert checkpoint. A separate [trained pole demonstration checkpoint](https://drive.google.com/file/d/1bd2CDBAJr7fV_5Tpy5-2hrq3ZZIyg3so/view?usp=sharing) is available for playback; course models are not included in Git.

Training and demonstrations require the repository's `3rdparty/SuperMimic` submodule to be initialized; there is no need to clone the upstream repository separately. The source robot model, route data, and training configuration snapshots are maintained in this repository.

The baseline reproduces the three-pole route from source task 7.7 in the repository's soccer-field scene. It uses task-specific observations, rewards, reset and termination conditions, and checkpoint format. For compatibility details, asset paths, SHA-256 hashes, data splits, and full reproduction commands, see the [asset and reproduction guide](../../docs/pole-weaving.md) (currently in Chinese).

## Quick Start

Run these commands from the repository root. Before training, download the [task 7.7 warmup checkpoint](https://drive.google.com/file/d/1PmM1NoD09gKIwzqlj3-rCeQRdffVutjJ/view?usp=sharing) and register it at the path shown in the [asset and reproduction guide](../../docs/pole-weaving.md). The smoke run performs one update and writes `latest.pt` for checking model loading and headless playback; it is not intended to train a policy that meets task metrics. The model is saved under `logs/q1-pole-weaving/run-<timestamp>/latest.pt`; a full run usually also produces `best.pt`. Evaluation and playback do not require the warmup file.

```bash
# Read-only preflight: hashes data and source assets, strictly loads the warmup checkpoint,
# parses the source configuration, and checks hardware. Does not create a run directory.
python -m pole_task.training --preflight-only --config courses/pole-weaving/pole_task/config.json

# Short smoke training: performs one update and writes latest.pt for playback.
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json --max-updates 1

# Train this project's pole-weaving configuration (2,000 warmup + 500 Actor updates).
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json

# Resume a run from this project.
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt

# Headless evaluation, full acceptance checks, and playback.
# Supports only pole-weaving checkpoints from this project.
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01

# Play five times; the random seed increments from 20261103.
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt \
  --case nominal-01 --gui --seed 20261103 --loops 5
```

The default output directory is `logs/q1-pole-weaving/`, separate from the soccer configuration's `logs/auto`. Pole-weaving training and playback do not load soccer policy or checkpoint code.

## Headless Demonstration

To play the provided trained model without running training, download the [pole demonstration checkpoint](https://drive.google.com/file/d/1bd2CDBAJr7fV_5Tpy5-2hrq3ZZIyg3so/view?usp=sharing) and save it as `logs/q1-pole-weaving/pole-demo.pt`, then run:

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/pole-demo.pt \
  --case nominal-01 --device cpu
```

You can also use `latest.pt` from a short smoke run to run the complete `nominal-01` route and save demonstration results. Full training is not required:

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt \
  --case nominal-01 --device cpu
```

Course models are not included in the repository. After changing the algorithm, keep a separate training output directory and evaluate with the same task configuration.
