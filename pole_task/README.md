# Q1 杆间绕行

本配置在仓库的足球场景中复现源任务 7.7 的三杆 S 形路线。它使用与源任务兼容的
Q1 机器人和控制接口、每帧 78 维观测、10 帧历史、149 维路线信息、以机器人为基准的
目标，以及源任务的 Actor、世界模型和 Critic 架构。

训练严格使用 `pole_task/source_manifest.json` 中登记的任务 7.7 warmup checkpoint 初始化，
不会改用足球专家权重或源任务最终 Actor。无需检出上游 `SuperMimic`：源机器人模型、路线数据
和训练配置快照均由本仓库管理。从头训练需要 warmup checkpoint，可联系
Xiaofeng Kuang（`stevenkwong@qq.com`）获取。文件路径、SHA-256、数据划分和完整命令见
[../docs/pole-weaving.md](../docs/pole-weaving.md)。使用本项目训练得到的 checkpoint
进行评估或回放时，不需要该 warmup 文件。

## 快速开始

```bash
# 只读检查：计算数据和源资产哈希，严格加载 warmup，解析源配置并检查硬件；不会创建运行目录。
python -m pole_task.training --preflight-only --config pole_task/config.json

# 训练本项目的杆间绕行配置（2,000 次 warmup + 500 次 Actor 更新）。
python -m pole_task.training --config pole_task/config.json

# 恢复本项目训练的运行任务。
python -m pole_task.training --config pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt

# 无界面评估、完整验收和回放（仅支持本项目杆间绕行 checkpoint）。
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01

# 连续播放 5 次；随机种子从 20261103 递增。
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt \
  --case nominal-01 --gui --seed 20261103 --loops 5
```

默认输出目录为 `logs/q1-pole-weaving/`，与足球配置的 `logs/auto` 分开。杆间绕行的训练和
回放不会加载足球策略或 checkpoint 代码。
