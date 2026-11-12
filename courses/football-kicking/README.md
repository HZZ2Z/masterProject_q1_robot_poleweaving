# Soccer Kicking

English | [简体中文](README.zh-CN.md)

This course provides a vision-based Q1 soccer policy training baseline. It covers expert controllers, trajectory data, imitation learning, and online policy updates, and supports experimentation with observations, training configurations, and policies.

![Soccer kicking policy demonstration](../../docs/assets/demos/football-kicking.gif)

## First Run

Set up the environment using the [shared environment setup guide](../../docs/setup.md) (currently in Chinese). Smoke training requires the [soccer expert checkpoint](https://drive.google.com/file/d/1Hb_vHlk4TK_Y8A7NvUUSi5kduLvhXu7P/view?usp=sharing); full training also requires the complete successful-trajectory dataset. These assets are not distributed in Git. Download the expert weights to `shared/supermimic/expert/checkpoints/model_17800.pth` before running the smoke workflow below. See [Preparing Resources](#preparing-resources) for details.

Run these commands from the repository root. First collect four short training episodes with the teacher model, then start a small training run:

```bash
python -m expert.collect --bounded-targets --episodes 4 --seed 100008 \
  --threads 1 --checkpoint shared/supermimic/expert/checkpoints/model_17800.pth \
  --output logs/football-kicking-smoke-data
python -m q1_soccer.train --config courses/football-kicking/configs/smoke.json
```

Smoke training checks the pipeline and produces a loadable checkpoint; it does not mean the policy has reached usable performance. To run a demonstration with the latest training output:

```bash
python -m q1_soccer.play --checkpoint logs/latest/last.pt --headless --steps 1000
```

## Preparing Resources

Full training uses the registered complete successful-trajectory dataset. Expert weights and data are not tracked by Git:

- Expert weights: `shared/supermimic/expert/checkpoints/model_17800.pth`
- Training data: `shared/supermimic/expert/datasets/q1_bounded_v1/`
- Held-out validation set: located in the dataset directory above

Download the [soccer expert checkpoint](https://drive.google.com/file/d/1Hb_vHlk4TK_Y8A7NvUUSi5kduLvhXu7P/view?usp=sharing) and save it as `shared/supermimic/expert/checkpoints/model_17800.pth`; the collection and training commands load it from that path. The full successful-trajectory dataset is a separate resource: if it is missing locally, follow the [SuperMimic expert resource guide](../../shared/supermimic/expert/README.md) to obtain it. With only the expert weights, you can collect smoke data using the command above. The output directory must be empty; use a new directory for each rerun and update the `dataset` path in `configs/smoke.json` accordingly.

## Full Training and Checks

After preparing the full dataset and teacher model, check the configuration and start training from the repository root:

```bash
python -m q1_soccer.train --config courses/football-kicking/configs/soccer.json --check
python -m q1_soccer.train --config courses/football-kicking/configs/soccer.json
```

Full training uses CUDA. Its configuration is `configs/soccer.json`; the smoke configuration uses CPU and is `configs/smoke.json`. Outputs include `best.pt`, `last.pt`, validation metrics, and the run configuration under `logs/q1-robot-supermimic-<timestamp>/`. `logs/latest/last.pt` points to the most recent run. See [baseline.json](baseline.json) for the recorded baseline.

Run the built-in wiring, checkpoint round-trip, and short-rollout checks for local evaluation:

```bash
python -m q1_soccer.verify --config courses/football-kicking/configs/smoke.json \
  --device cpu --smoke --report logs/football-kicking-evaluation.json
```

For an interactive demonstration, omit `--headless --steps`. Check the checkpoint path and number of steps in the output to record and compare experiments.
