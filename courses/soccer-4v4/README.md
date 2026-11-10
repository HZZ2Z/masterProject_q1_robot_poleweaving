# 4v4 足球对抗

本课程提供从基础步态到团队足球对抗的分阶段强化学习基线，支持调整策略网络、课程调度和奖励设计。课程阶段见[课程表](docs/CURRICULUM.md)。

![4v4 足球对抗演示](../../docs/assets/demos/soccer-4v4-match.gif)

## 首次体验

先按[统一环境搭建指南](../../docs/setup.md)安装环境。无需预训练模型即可预览基础场景；该 Warp 场景预览需要可用的 NVIDIA GPU：

```bash
python courses/soccer-4v4/play.py --neutral --headless --steps 30 --device cuda:0
```

如果暂时没有可用的 CUDA 设备，可以先运行 CPU 冒烟训练，检查训练管线能否完成一次更新。该步骤不生成可用策略，也不能用于判断策略效果：

```bash
python courses/soccer-4v4/train.py --backend cpu --device cpu --no-wandb \
  --num-envs 2 --horizon 8 --iterations 1 --epochs 1 \
  --stage balance --episode-steps 4 --job-type smoke \
  --output logs/soccer-4v4-smoke
```

## 训练

正式训练使用 Warp 后端、NVIDIA GPU 和更大规模的并行环境：

```bash
python courses/soccer-4v4/train.py --backend warp --device cuda:0 --no-wandb \
  --num-envs 4096 --iterations 20000 --stage balance \
  --output logs/soccer-4v4
```

训练默认写出 `last.pt`，并按课程门控保存阶段检查点。训练环境、奖励与课程阶段说明见[课程表](docs/CURRICULUM.md)。仓库不包含预训练的 4v4 模型。

## 演示与评估

完成 CPU 冒烟或正式训练后，可使用生成的 checkpoint 加载策略。策略演示使用 Warp 环境，需要 NVIDIA GPU：

```bash
python courses/soccer-4v4/play.py --checkpoint logs/soccer-4v4-smoke/last.pt \
  --course balance --headless --steps 30 --device cuda:0
```

正式模型路径为对应训练输出目录中的 `last.pt` 或阶段检查点。训练日志记录课程成功率、稳定率和对抗指标；切换课程阶段后，可用相同方法检查模型行为并记录 checkpoint 路径、课程阶段和运行步数：

```bash
python courses/soccer-4v4/play.py --checkpoint logs/soccer-4v4/last.pt \
  --course match --headless --steps 300 --device cuda:0
```
