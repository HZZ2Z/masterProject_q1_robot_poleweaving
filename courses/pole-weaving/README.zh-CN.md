# S 型绕杆

[English](README.md) | 简体中文

本课程提供基于 SuperMimic 世界模型框架的 Q1 三杆 S 型绕行基线，涵盖任务环境、观测与奖励设计，可用于复现基线和改进策略。

![S 型绕杆策略演示](../../docs/assets/demos/pole-weaving.gif)

## 首次体验前

先按[统一环境搭建指南](../../docs/setup.md)完成安装。训练配置默认使用 `cuda:0`，需要可用的 NVIDIA GPU 和至少 4 GB 空闲显存；评估和演示默认在 CPU 上运行。训练需要任务 7.7 warmup checkpoint；请下载[绕杆 warmup checkpoint](https://drive.google.com/file/d/1PmM1NoD09gKIwzqlj3-rCeQRdffVutjJ/view?usp=sharing)，并按[资产与复现说明](../../docs/pole-weaving.md)登记到指定路径。该权重不能用足球专家权重替代。另有可直接回放的[绕杆演示策略 checkpoint](https://drive.google.com/file/d/1bd2CDBAJr7fV_5Tpy5-2hrq3ZZIyg3so/view?usp=sharing)；课程模型不纳入 Git。

训练和演示需要初始化仓库自带的 `3rdparty/SuperMimic` 子模块，无需另行克隆上游仓库。源机器人模型、路线数据和训练配置快照均由本仓库管理。

基线在仓库足球场景中复现源任务 7.7 的三杆路线，采用任务专属的观测、奖励、重置、终止条件和 checkpoint 格式。任务与源实现的兼容细节、资产路径、SHA-256、数据划分和完整复现命令见[资产与复现说明](../../docs/pole-weaving.md)。

## 快速开始

以下命令从仓库根目录执行。训练前请下载[任务 7.7 warmup checkpoint](https://drive.google.com/file/d/1PmM1NoD09gKIwzqlj3-rCeQRdffVutjJ/view?usp=sharing)，并按[资产与复现说明](../../docs/pole-weaving.md)放到指定路径。冒烟运行执行一次更新，生成可用于检查模型加载和无头演示的 `latest.pt`，不要求训练出达到任务指标的策略。模型位于 `logs/q1-pole-weaving/run-<timestamp>/latest.pt`；正式运行通常还会生成 `best.pt`。评估和演示不需要 warmup 文件。

```bash
# 只读检查：计算数据和源资产哈希，严格加载 warmup，解析源配置并检查硬件；不会创建运行目录。
python -m pole_task.training --preflight-only --config courses/pole-weaving/pole_task/config.json

# 短程冒烟训练：执行 1 次训练更新并写出可用于演示的 latest.pt。
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json --max-updates 1

# 训练本项目的杆间绕行配置（2,000 次 warmup + 500 次 Actor 更新）。
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json

# 恢复本项目训练的运行任务。
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json \
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

默认输出目录为 `logs/q1-pole-weaving/`，与足球配置的 `logs/auto` 分开。杆间绕行的训练和回放不会加载足球策略或 checkpoint 代码。

## 无头演示

如需直接回放提供的训练策略，下载[绕杆演示 checkpoint](https://drive.google.com/file/d/1bd2CDBAJr7fV_5Tpy5-2hrq3ZZIyg3so/view?usp=sharing)，保存为 `logs/q1-pole-weaving/pole-demo.pt`，然后运行：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/pole-demo.pt \
  --case nominal-01 --device cpu
```

也可以使用短程冒烟训练生成的 `latest.pt`，运行完整的 `nominal-01` 路线并写出演示结果；无需先完成正式训练：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt \
  --case nominal-01 --device cpu
```

课程模型不随代码提交。修改算法后，请保留独立的训练输出目录，并使用相同的任务配置进行评估。
