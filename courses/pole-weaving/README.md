# S 型绕杆

本课程以现有杆间控制策略作为可复现基线。学生可从任务环境、观测与奖励入手，独立调整策略和训练方法。

使用 SuperMimic 世界模型框架实现的杆间任务基线，复现三杆 S 型路线。任务拥有独立的观测、奖励、重置、终止条件和 checkpoint 格式。

本配置在仓库的足球场景中复现源任务 7.7 的三杆 S 形路线。它使用与源任务兼容的
Q1 机器人和控制接口、每帧 78 维观测、10 帧历史、149 维路线信息、以机器人为基准的
目标，以及源任务的 Actor、世界模型和 Critic 架构。

训练严格使用 `pole_task/source_manifest.json` 中登记的任务 7.7 warmup checkpoint 初始化，
不会改用足球专家权重或源任务最终 Actor。训练和演示需要初始化仓库自带的
`3rdparty/SuperMimic` 子模块；无需另行克隆上游仓库。源机器人模型、路线数据
和训练配置快照均由本仓库管理。从头训练需要 warmup checkpoint，可联系
Xiaofeng Kuang（`stevenkwong@qq.com`）获取。文件路径、SHA-256、数据划分和完整命令见
[../../docs/pole-weaving.md](../../docs/pole-weaving.md)。使用本项目训练得到的 checkpoint
进行评估或回放时，不需要该 warmup 文件。

## 快速开始

训练前需要维护者提供并登记的任务 7.7 warmup checkpoint；本课程不会用足球策略权重替代它。训练成功后，冒烟运行产生的 `latest.pt` 即可检查模型加载和无头演示，无需训练出达到任务指标的完整策略。模型位于 `logs/q1-pole-weaving/run-<timestamp>/latest.pt`；正式运行通常还会生成 `best.pt`。模型评估和演示不再需要 warmup 文件。详细资产清单见[维护说明](../../docs/pole-weaving.md)。

```bash
# 只读检查：计算数据和源资产哈希，严格加载 warmup，解析源配置并检查硬件；不会创建运行目录。
python -m pole_task.training --preflight-only --config pole_task/config.json

# 短程冒烟训练：执行 1 次训练更新并写出可用于演示的 latest.pt。
python -m pole_task.training --config pole_task/config.json --max-updates 1

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

## 无头演示

使用短程冒烟训练生成的 `latest.pt`，运行完整的 `nominal-01` 路线并写出演示结果；无需先完成正式训练：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt \
  --case nominal-01 --device cpu
```

课程模型不随代码提交。修改算法后，请保留独立的训练输出目录，并使用相同的任务配置进行评估。
