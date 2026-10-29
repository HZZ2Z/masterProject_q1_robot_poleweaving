# Q1 Humanoid Kick / SuperMimic

Q1 V3（22 自由度）的视觉踢球仿真训练。GPU 物理后端为 **MuJoCo Warp**，CPU 用原生 MuJoCo Batch 做参考验证。新项目与 `../q1-humanoid-kick` 的运行进程、日志、checkpoint 独立。

当前状态：旧版失败 run 已停止；修正版 v3 已通过真实回合对照并重新启动，**尚未证明踢球收敛**。默认命令必须显式选择 `--preflight`、`--smoke` 或 `--start-training`，防止误启动正式训练。

## 方法与上游关系

[PrimebotTech/SuperMimic](https://github.com/PrimebotTech/SuperMimic) 克隆在 `3rdparty/SuperMimic`，固定提交见 `SOURCE.json`。上游是 Q1 V2 全身动作跟踪，不是足球任务。这里复用上游 `supermimic.models.mlp` 和 `frozen`，实现其自回归世界模型与短视野解析策略梯度的足球适配；不直接调用上游 `SuperMimic` 类或跟踪训练器，也不宣称是论文完整复现。

- 真实采样：Q1 V3、足球接触、球门、虚拟相机噪声/延迟、22 关节 PD 和域随机化。物理 2ms、控制 20ms，CUDA stream 共享和物理 CUDA Graph 沿用已验证的 Q1 后端。
- 世界模型：预测归一化增量状态、缩放后的足球奖励和摔倒概率；多步自回归监督。状态前 79 维为策略观测，其后包含 critic 特权状态、机器人/球 qpos、qvel、延迟 PD 目标和增益、视觉延迟缓冲、质量/摩擦和课程进度等。
- 策略：仅输入前 79 维视觉/本体观测，不读取特权坐标。穿过冻结世界模型和 critic 的想象轨迹传播梯度；critic 使用带存活概率的 lambda return。默认想象 16 步。
- 任务适配：上游基于动作参考的解析奖励改成从真实足球奖励监督的可微奖励头；没有 PPO 或 AMP 判别器。已有 Q1 动作仅供课程阶段的参考状态初始化使用。
- 每个终止目标在自动重置前采集；时限截断与摔倒分开标注，自回归监督不会穿越重置边界。成功要求进球后机器人仍未跌倒，环境随机给球加速已关闭。
- 站立到足球的自适应课程沿用 Q1 任务。模型先收集 warmup 标定固定归一化，默认前 19 轮只拟合世界模型，从第 20 轮开始策略更新。

这是需要实训检验的模型式足球基线。奖励头和视觉状态预测会存在模型误差，短测通过不说明能稳定站立或踢球。策略没有历史编码器；当前动力学输入也未覆盖全部物理随机量，应在正式训练时评估长时预测和部分可观测性的影响。

## 环境与检查

使用机器已有的 `q1-kick` 环境：Python 3.11、Torch 2.9、MuJoCo/MuJoCo Warp 3.17、Warp 1.18。这里仅导入兼容的上游网络组件，**不要直接安装上游 pyproject/uv.lock**，它要求另一套 Python 3.13 / MuJoCo 3.6 依赖。

```bash
conda activate q1-kick
cd /home/kemove/workspace/Q1Robot/q1-humanoid-kick-SuperMimic
python train.py --preflight --device cpu
python train.py --preflight --device cuda:0 --num-envs 2
python train.py --smoke --output logs/smoke
python -m pytest -q
```

`--preflight` 仅做 4 个控制步，无学习；`--smoke` 默认 CPU，固定 2 环境、2 轮、微型网络，关闭 W&B；第 2 轮切入足球反馈以覆盖站立和踢球两阶段。可显式加 `--device cuda:0` 检查 GPU 上的学习更新。测试包含一次小批量 GPU 的 6 秒仿真，无长训练。如终端继承了 AppImage 的 Python/库环境变量，可使用：

```bash
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/kemove/miniconda3/envs/q1-kick/bin/python train.py --smoke
```

资源、参考动作和环境工具是旧项目的独立副本，保留原许可证与资源来源清单；`SOURCE.json` 记录依赖来源。依赖见 `requirements.txt`，Torch 使用现有 CUDA 对应安装。

## 待用户要求后才启动

正式配置在 `configs/soccer.yaml`，默认 1024 环境，先作为模型式训练起点，没有为当前显卡余量做吞吐调参。1024 环境完整更新短测约占 5.1GB 显存，Torch allocator 限额为总显存的 30%（Warp 内存另计），为并行 PPO 留出余量。

```bash
# 只有用户明确要求起跑后才执行
python train.py --start-training --config configs/soccer.yaml
```

W&B 使用 `mohismlab-robotics/q1-humanoid-kick-SuperMimic`，从用户目录已有登录或 `WANDB_API_KEY` 环境变量读取凭据，不在仓库保存 key。`--no-wandb` 可关闭云端记录。

输出：`logs/q1_supermimic/<时间>/` 内含配置、`metrics.jsonl`、TensorBoard 和 `checkpoint_<轮次>.pt`、`last.pt`。每轮记录世界模型/actor/critic loss、奖励、存活与课程进度；仅在有回合结束时记录成功/摔倒率。SIGINT/SIGTERM 会在当前迭代完成后保存。

```bash
python train.py --start-training --checkpoint <checkpoint.pt> --iterations 20000
python play.py --checkpoint <checkpoint.pt> --steps 1000
python play.py --checkpoint <checkpoint.pt> --viewer
```

`iterations` 是累计轮数。恢复网络、三个优化器、归一化、课程状态与 RNG；重建仿真和回放池，不保证轨迹逐位复现。仅加载可信 checkpoint。CPU 回放用于可视化与功能验证，不能替代固定种子的 GPU 独立评测。

## 2026-10-30 任务同步

已同步 `q1-humanoid-kick` 当前奖励 **version 4** 和近球起步课程：

- 正踢权重 2、侧踢权重 8；要求足端接近、站稳、足球真实朝球门移动，不给空踢奖励。
- 近球停滞惩罚 -2，带准备宽限和阈值迟滞；远球停滞按 2 秒窗口内的净接近进度判断。
- 足球在球场中心半径 3.5 米圆盘内均匀出生；近球起步保留足球位置，调整机器人位置和朝向。当前采用下述固定 25% / 25% / 50% 配额，从首次重置起生效，不自动降低比例或扩大间隙。
- 世界模型包含近/远球计时器、迟滞状态、接近窗口锚点和近球起步标记。每轮同时记录 `reward_terms/*`、`soccer/*` 和 `curriculum/near_start_*`。
- 本项目不使用 AMP 判别器，因此原 PPO 项目中“停滞时削减 AMP”的乘子不产生额外训练奖励；真实足球惩罚和踢球奖励直接监督世界模型。

Checkpoint 格式升级为 v2，保存任务指纹、环境配置和近球课程统计；旧版短测 checkpoint、旧奖励版本和旧 PPO checkpoint 不可直接续训，必须新建本项目 run。每个 run 保存 `environment.yaml`，用于确认实际奖励/课程口径。

正式训练仍需用户明确要求后，通过 `--start-training` 启动。可用下面的入口清理终端继承的 AppImage 环境变量（不传参数只显示参数错误，不会起跑）：

```bash
./scripts/run.sh --preflight --device cuda:0
./scripts/run.sh --smoke --device cuda:0 --output logs/gpu_smoke
# 以下命令仅待用户明确要求起跑后执行
./scripts/run.sh --start-training --config configs/soccer.yaml
```

## 固定混合课程、显示和日志（2026-10-30）

1024 环境固定分为 256 / 256 / 512 三组，局部重置保持组别：

| 比例 | 初始条件 |
|---|---|
| 25% | 机器人和球在门线内场地均匀采样，根位置/球心 x ∈ [-6.5,6.5]，y ∈ [-4,4]，拒绝机器人与球的初始重叠 |
| 25% | 机器人和球在中心半径 3.5m 圆内按面积均匀采样，拒绝初始重叠 |
| 50% | 球放在圆内，机器人以中立安全姿态朝向球；求解实际足端包围盒距离，使最近脚球间隙均匀分布在 10–25cm，初始速度为零 |

正式环境数要求为 4 的倍数以保证配额精确。微型短测/单环境回放按最大余数取整。固定混合覆盖先前的近球比例递减课程，站立到踢球的奖励课程仍独立运行。

场景 XML、草皮/足球纹理、场线、默认回放视角和 F 键视锥开关已与 PPO 对齐；`logs/visual_check/` 下保存同状态渲染对比，最大像素差为 0。机器人惯量、关节、执行器和碰撞几何参数经对比保持不变。

W&B 与 PPO 共用主要指标名与单位：`reward`（已结束回合总回报均值）、`success_rate`、`fall_rate`、`steps`、`episode/duration_s`、`episode/<奖励项>`、`episode/ball_forward_distance`、`stability/*`、`curriculum/*` 和 `performance/*`。仅对已结束回合统计成功/摔倒率，不填虚假零值；另记每控制步的 `reward_terms/*` 和 `reward/control_step`。世界模型/actor/critic loss 在 `loss/*`。

两种算法的总回报成分有所不同：PPO 会加入 AMP，当前模型式训练没有 AMP；W&B 配置标记 `reward_includes_amp=false`。共同的足球奖励分项、成功/摔倒判据与回合时长可按同一含义比较。

当前启动进程与日志路径见 `logs/active_training.json`。正式训练从独立随机初始化开始，不加载 PPO 权重。

## 首次正式训练诊断

`2026-10-30-00-59-57-739042` 在第 2303 轮正常停止并保存。最近回合约 0.23 秒，速度阈值终止为主。相同种子、8 环境、6 秒 CPU 参考对照：零动作 9 次摔倒，0.08 标准差噪声 22 次，checkpoint_2300 策略 191 次；策略约 38.8% 动作维度绝对值超过 0.95。记录见 `logs/fall_diagnosis.json`。

目前只对世界模型做 4 步监督，策略在 16 步预测中优化学习奖励头，critic 也只使用模型想象目标；没有真实仿真改善门槛或策略信任约束。上述设计需要修正并进行真实轨迹对照，不能用 loss 有限或模型拟合误差下降代替训练有效性证明。

## 学习目标修正版 v3

第一次失败 run 不再续训。v3 从零初始化，并改变了训练约束：

- 世界模型在 8 步真实转移上自回归监督；策略想象最多 4 步。1/8 环境专用于模型验证，不进入模型训练采样。
- 策略优化明确的平滑物理目标（生存、机身高度/倾斜、站姿、动作变化、根加速度、随课程启用的接近球/球门进度），不使用学习奖励头或学习终止头的梯度。完整 reward v4 仍用于真实采样、TD 和 W&B 主回报；平滑规划目标不是完整接触奖励的逐项等价实现。
- Critic 用真实转移的 reward/终止标记进行 TD 学习，不再用模型想象回报自我训练。
- 想象中保持质量、摩擦、PD 增益、默认姿态和课程组别不变，并强制动作历史与四元数一致。
- 站立阶段动作限制在 ±0.10，探索标准差 0.01；动作范围随足球课程进度增加。单次 actor 更新在真实观测上的动作变化另有限制。
- 仅在验证物理误差低于门槛、且模型优于“状态不变”预测后更新策略。`learning/model_ready` 与 `model_validation/*` 明确显示是否满足门槛。
- 每 50 轮独立运行固定种子、32 个原生 MuJoCo 环境、6 秒真实对照。记录第一回合的生存时长/回报，不通过任意重置刷样本。比中立站姿基线或此前接受策略退化时，恢复上一次接受的 actor 与优化器状态。
- `evaluation/*` 报告真实验证结果、是否接受和回滚次数；`policy/*` 报告实际动作幅度与饱和比例。训练中的 `fall_rate` 仍是已结束回合中的摔倒比例，初期应结合回合时长与 5 秒存活率看。

Checkpoint 格式为 v3，增加真实验证的接受策略/优化器状态；旧版失败 run 不可恢复到 v3。场景、reward v4、固定 25/25/50 混合以及 W&B 项目保持不变。真实验证门槛用于防止已发现的策略崩溃，不代表已证明足球收敛。


v3 的实际 actor 目标为真实轨迹策略梯度加 0.01 倍物理模型规划辅助项（没有 PPO clip）。采样使用带固定探索尺度的 tanh 高斯，策略梯度使用当前迭代的 on-policy 状态、执行动作、真实 reward、真实终止标记和真实 TD advantage；不使用回放池的旧动作作 on-policy 梯度。模型准备门槛按完整 4 步验证：高度 MAE <1cm、关节 RMSE <0.025rad、根速度 RMSE <0.5，同时优于状态保持基线至少 20%。`evaluation/candidate_*` 记录候选，`evaluation/*` 记录接受或回滚后重新测量的实际策略。


修正版第 300 轮固定评估（32 场景，6 秒）：零动作 5 秒存活率 50%、摔倒比例 50%、平均回报 1.179；修正策略对应 68.75%、31.25%、7.339。第 250 轮的一次不良候选已被回滚；这些是实测接受检查，不是长期收敛成绩。正式 run 从修正版 `checkpoint_300.pt` 继续，旧版 `checkpoint_2303.pt` 未加载。当前进程与日志见 `logs/active_training.json`。


### 2026-10-30：近球优先课程与统计窗口

从原训练 checkpoint_3206.pt 通过 `--migrate-curriculum` 续训；该选项只允许课程变化，仍检查学习器版本、状态维度、奖励、控制、初始状态、随机化和资产。保留网络、优化器、归一化及课程历史；物理轨迹、replay 与回合统计窗口重新开始。

保持 25% 场内、25% 半径 3.5 m 圆内均匀面积采样、50% 朝向球且脚球间隙 10–25 cm。起步门槛从 80% 改为 65%（仍需达到原最早迭代数，同时检查活跃环境与重置队列 5 秒存活率）。释放时足球进度为 0.02；进度不超过 0.2 时，仅近球组获得足球奖励，其余组继续站立；之后其余组进度按 (progress-0.2)/0.8 增长。参考动作初始化跟随全场进度。全体动作上限仍随总进度从 0.1 逐步增大，保留真实物理策略验收。此调整启用学习信号，并不表示策略已学会踢球。

主 `fall_rate`、`success_rate`、`reward`、`steps` 和 `episode/*` 使用最近 1000 个已结束回合，逐回合加权。`instant/*` 保留单轮数据；`statistics/window_episodes`、`window_capacity`、`window_ready`、`new_completed_episodes` 和 `window_simulated_seconds` 说明样本量与累计模拟时长。新运行窗口重新计数，不沿用旧课程结果。窗口内仍只含结束回合，不能当成所有活跃环境的失败概率；应同时查看 `stability/reset_cohort_survival_5s` 和固定 6 秒 `evaluation/*`。

验证：54 项测试通过；CUDA checkpoint 迁移及两轮学习 smoke 通过。正式训练使用原 3206 checkpoint，不使用 smoke 更新后的权重。W&B 新运行 pz6y2g96。

### 2026-10-30：踢球停滞修正与足球 best

从第 4586 轮存档续训。近球组足球奖励不再被 0–6% 的动作课程进度稀释：释放站立阶段后该组任务强度为 1；动作幅度继续按课程逐步增加。课程推进依据重置队列 5 秒存活率（推进 60%，退回 45%），不再受活跃环境同步重置影响。近球进球率未达到 2% 时，动作课程进度最多 0.35，全场足球进度为零；外力扰动跟随全场进度。25/25/50 重置分布保持不变。

站立收入在任务组持续停滞时逐步降为零，近球接近奖励乘 10，停滞惩罚乘 0.5；近球练习回合上限 12 秒，超时不算摔倒。基础 v4 奖励项/场景保持，但课程权重已改变，因此总奖励不能直接与旧运行比较。探索标准差改为 0.02 + 0.12 * 动作课程进度。

固定种子真实 MuJoCo 验收使用 32 个环境、6 秒首回合及固定足球任务，记录 near_approach_m、near_kick_event_rate、near_net_goal_progress_m、near_success_rate。kick_event 使用与踢球奖励相同的脚球间隙、脚速、朝门球速、直立且未摔倒条件，是有效踢球事件判据，不是独立接触传感器。净向门推进使用起终点球门距离差，避免累加来回抖动的正向距离。验收先要求至少 65% 的 5 秒存活率、失败率不超过 35%，且不低于中立动作存活基准，再比较足球分数；站立总奖励不能压过足球表现。更改前的验收分数重新计算。

`play.py --best` 现在只加载存档内 soccer_best_actor：必须通过安全底线，出现踢球事件且近球组平均净向门推进至少 1 cm。尚无此权重时明确报错，不将站立策略冒充足球 best。`--accepted` 可查看最近通过验收的策略。两者均恢复对应动作上限；GUI 仍需重启加载新存档。

验证：57 项测试通过，包含同步重置课程推进、停滞收入、12 秒非摔倒超时、足球优先策略选择和不安全策略拒绝；CUDA 两轮存档续训 smoke 通过。正式训练从原第 4586 轮继续，不使用 smoke 更新后的权重。
