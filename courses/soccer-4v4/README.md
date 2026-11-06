# 4v4 足球对抗

本课程提供从基础步态到团队对抗的分阶段强化学习基线。训练环境、奖励、课程表和策略实现位于本目录；学生可基于该基线尝试新的策略网络、课程调度或奖励设计。详细课程阶段见 [课程表](docs/CURRICULUM.md)。

## 环境

按仓库根目录说明创建统一环境。4v4 训练需要 PyTorch、MuJoCo、MuJoCo Warp、Warp 和本课程随附的 MAPPO 依赖。常用 Q1 模型和动作资源由 `shared/q1-mujoco/` 提供。

## 短程训练

下面的 CPU 命令完成一次小规模更新，用于检查训练管线：

```bash
python train.py --backend cpu --device cpu --no-wandb \
  --num-envs 2 --horizon 8 --iterations 1 --epochs 1 \
  --stage balance --episode-steps 4 --job-type smoke \
  --output logs/soccer-4v4-smoke
```

正式训练使用 Warp 后端并增加并行环境数与迭代次数：

```bash
python train.py --backend warp --device cuda:0 --no-wandb \
  --num-envs 4096 --iterations 20000 --stage balance \
  --output logs/soccer-4v4
```

训练默认写出 `last.pt`，并按课程门控保存阶段检查点。对抗、策略共享和并行课程的详细说明见课程表。

## 无头演示

使用短程训练或正式训练生成的 checkpoint：

```bash
python play.py --checkpoint logs/soccer-4v4-smoke/last.pt \
  --course balance --headless --steps 30 --device cuda:0
```

正式模型路径为对应训练输出目录中的 `last.pt` 或阶段检查点。仓库不包含预训练的 4v4 模型。

## 评估

训练日志记录课程成功率、稳定率和对抗指标；根据需要在课程切换后使用 headless 演示检查模型运行，并保存检查点路径、课程阶段和运行步数。`play.py` 的演示可重复执行，适合比较学生迭代前后的行为：

```bash
python play.py --checkpoint logs/soccer-4v4/last.pt \
  --course match --headless --steps 300
```
