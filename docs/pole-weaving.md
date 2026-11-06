# Q1 杆间绕行

杆间绕行任务是独立的 SuperMimic 配置，用于在本仓库的足球场景中复现源任务 7.7 的三杆
S 形路线。它有独立且与源任务兼容的观测、奖励、重置、终止条件、Actor、世界模型和 Critic
接口；训练运行目录位于 `logs/q1-pole-weaving/`，checkpoint 格式也由本项目单独管理。
球和球门仅作为可见场景元素，不会进入杆间绕行任务的观测、奖励、重置或终止条件。

## 所需资产

杆间绕行任务不再需要旁边的 `../../SuperMimic` 仓库副本。源 Q1 XML 和训练配置快照已保存在
本仓库中；XML 使用 `expert/resources` 下已有的机器人网格。路线参考数据保存在
`pole_task/data/references/`。`pole_task/data/manifest.json` 中的绝对路径字段 `source`、
`source_catalog` 和 `source_geometry` 用于记录原始位置；程序运行时改用相对于仓库的 `file` 字段。

只有从源 warmup 开始训练时才需要外部提供模型文件。请联系 Xiaofeng Kuang（`stevenkwong@qq.com`）
获取 `checkpoint_2000.pt`，然后将其复制到登记路径：

```bash
mkdir -p pole_task/assets/checkpoints
cp /path/to/checkpoint_2000.pt pole_task/assets/checkpoints/source-task-7.7-warmup.pt
sha256sum pole_task/assets/checkpoints/source-task-7.7-warmup.pt
```

文件的 SHA-256 必须为
`b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5`.
训练前的预检查会核对该哈希。模型文件已加入 Git 忽略列表。源任务的最终 checkpoint 仅以哈希
记录来源，不需要复制到新主机。

`pole_task/source_manifest.json` 记录源代码版本和迁移文件的哈希。标记为
`required_for: ["provenance"]` 的文件仅作为历史来源记录，不需要原始仓库副本：

| 资产 | 登记路径 | 用途 |
| --- | --- | --- |
| `warmup_checkpoint` | `pole_task/assets/checkpoints/source-task-7.7-warmup.pt` | 严格用于训练初始化；通过邮件索取 |
| `passed_final_checkpoint_reference` | 源任务 7.7 最终 checkpoint 的哈希 | 仅记录来源；不需要该文件 |
| `warmup_config`, `actor_config` | `pole_task/data/source-configs/` | 本地训练配置快照 |
| `robot_xml` | `expert/resources/q1_description/source_task_7_7/q1_v3_mc.xml` | 本地源机器人模型 |
| 路线目录、几何数据和参考轨迹 | `pole_task/data/` | 本地任务输入 |
| 源模型、路线和训练文件 | manifest 中记录的上游版本和哈希 | 仅记录来源；不需要原仓库副本 |

程序先核对登记的 warmup 文件哈希，再通过 `strict=True` 加载 Actor、世界模型、Critic 和机器人的
全部张量，并检查源任务测试样例的一致性。如果 warmup 权重缺失、损坏或无法识别，预检查会指出
不匹配的字段并停止；不会创建运行目录，也不会改用其他 checkpoint。源任务最终 Actor 和足球专家
checkpoint 都不会作为替代权重。本项目训练得到的 checkpoint 可在没有源 warmup 权重或上游仓库的
情况下进行评估和回放。

## 数据划分

`pole_task/data/manifest.json` 将 `nominal-01`（675 个控制步）和留出的 `eval-01`
（671 个控制步）与其有序杆位布局对应起来。留出的参考轨迹不会进入训练批次，只用于评估。
`straight`、`left_turn` 和 `right_turn` 是训练及固定步态保持测试使用的参考轨迹。每条参考轨迹和
路线布局都记录了对应哈希。

## 预检查与训练

预检查为只读操作：它会验证数据对应关系及哈希、源文件清单、warmup 严格加载结果、解析出的
2,000 + 500 次更新配置、输出目录隔离情况和硬件状态。预检查不会打开查看器或创建运行目录。

```bash
python -m pole_task.training --preflight-only --config pole_task/config.json
```

训练只会向杆间绕行配置的输出目录写入文件。每次运行都会记录源代码版本、warmup 路径和哈希、
解析后的训练阶段、优化器和学习率配置、数据划分、随机种子、环境步数预算、指标、停止原因及选用的
checkpoint：

```bash
python -m pole_task.training --config pole_task/config.json
```

训练配置固定为：先冻结 Actor，在确定性物理环境中进行 2,000 次 warmup 更新；再开启随机化并冻结
每帧历史中的 6 个杆相关输入，进行 500 次 Actor 更新。每次更新都会按登记的批次大小和预测时长拟合
世界模型，然后拟合 Critic；在 Actor 阶段，还会通过与源任务对齐的想象过程拟合 Actor。机器人实际
碰杆时，该回合会终止，并作为 replay 边界和失败轨迹处理。

恢复训练前会验证 checkpoint 格式、源 warmup 哈希、机器人和数据身份、配置及训练方案；验证通过后，
恢复本项目的 Actor、世界模型、Critic、优化器、更新序号、训练阶段和最佳分数：

```bash
python -m pole_task.training --config pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt
```

如需限制试运行时长，可添加 `--max-updates N`（不能超过登记的 2,500 次更新预算）。如果
`best.pt` 存在，就使用它；否则使用 `latest.pt`。程序不会搜索足球或源任务最终 checkpoint。

## 无界面评估与回放

以下命令默认以无界面模式运行，只接受本项目训练的杆间绕行 checkpoint。checkpoint 缺失、格式属于
足球任务、配置不匹配或属于源任务最终模型时，程序会拒绝加载，不会改用其他权重。

使用固定随机种子评估登记的常规、留出和步态保持用例：

```bash
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

运行完整的无界面验收：测试全部登记用例，以及 20 次使用固定随机种子的常规路线试跑。常规和留出
用例必须到达终点 0.5 m 范围内，根部 XY 均方根误差（RMSE）不超过 0.5 m，且不能碰杆或摔倒；
步态保持用例不能摔倒，RMSE 不超过 0.3 m；20 次常规路线试跑必须全部通过。

```bash
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

不打开显示窗口，回放一个登记用例：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01
```

仅在有可用显示器时添加 `--gui`。添加 `--loops N` 可连续播放 N 次，每次的随机种子依次为
`--seed`、`--seed + 1`，以此类推。例如：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt \
  --case nominal-01 --gui --seed 20261104 --loops 5
```

每次播放的路线完成情况、终点距离、运行步数、碰杆次数、摔倒情况、根部 XY RMSE 和成功状态，
都会写入 checkpoint 所在运行目录旁的结果文件。GUI 循环播放时会在同一个 MuJoCo 窗口内重置
仿真并开始下一轮，窗口两侧的工具栏默认关闭。
