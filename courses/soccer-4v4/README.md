# 4v4 Soccer

English | [简体中文](README.zh-CN.md)

This course provides a staged reinforcement-learning baseline from basic gait to team soccer. It supports experimentation with policy networks, curriculum scheduling, and reward design. See the [curriculum](docs/CURRICULUM.md) for the training stages.

![4v4 soccer demonstration](../../docs/assets/demos/soccer-4v4-match.gif)

## First Run

Set up the environment using the [shared environment setup guide](../../docs/setup.md) (currently in Chinese). You can preview the basic scene without a pretrained model. The Warp scene preview requires an NVIDIA GPU:

```bash
python courses/soccer-4v4/play.py --neutral --headless --steps 30 --device cuda:0
```

If CUDA is not available, run a CPU smoke training to check that the training pipeline completes one update. This does not produce a usable policy and cannot be used to judge policy performance:

```bash
python courses/soccer-4v4/train.py --backend cpu --device cpu --no-wandb \
  --num-envs 2 --horizon 8 --iterations 1 --epochs 1 \
  --stage balance --episode-steps 4 --job-type smoke \
  --output logs/soccer-4v4-smoke
```

## Training

Full training uses the Warp backend, an NVIDIA GPU, and a larger number of parallel environments:

```bash
python courses/soccer-4v4/train.py --backend warp --device cuda:0 --no-wandb \
  --num-envs 4096 --iterations 20000 --stage balance \
  --output logs/soccer-4v4
```

Training writes `last.pt` by default and saves stage checkpoints when curriculum gates are met. See the [curriculum](docs/CURRICULUM.md) for details on environments, rewards, and stages. The repository does not include a pretrained 4v4 model.

## Demonstration and Evaluation

After a smoke or full training run, load the generated checkpoint to run the policy. Policy demonstrations use the Warp environment and require an NVIDIA GPU:

```bash
python courses/soccer-4v4/play.py --checkpoint logs/soccer-4v4-smoke/last.pt \
  --course balance --headless --steps 30 --device cuda:0
```

For a full model, use `last.pt` or a stage checkpoint from the corresponding training output directory. Training logs record curriculum success rate, stability rate, and competition metrics. After changing curriculum stages, use the same approach to inspect behavior and record the checkpoint path, course stage, and number of steps:

```bash
python courses/soccer-4v4/play.py --checkpoint logs/soccer-4v4/last.pt \
  --course match --headless --steps 300 --device cuda:0
```
