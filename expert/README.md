# Q1 Expert

独立的 Q1 专家推理与轨迹采集工具。网络、配置、机器人资源、参考初始化动作已抽取到此目录，不导入相邻工程的代码。默认使用本地 `checkpoints/best.pth`，对应 `model_17800.pth`。

在 SuperMimic 仓库根目录运行：

```bash
# 专家回放；可加 --out-video expert/videos。
python -m expert.play

# 采集完整回合，保留成功、超时、出界和摔倒。
python -m expert.collect --bounded-targets --episodes 256 --seed 200000 \
  --output expert/datasets/new_batch

# 校验数据、力矩回放及教师动作。
python -m expert.check_dataset expert/datasets/new_batch
python -m expert.replay_check expert/datasets/new_batch
python -m expert.verify_teacher expert/datasets/new_batch
```

默认完整运行到专家原来的终止条件。`--max-control-steps` 仅用于有界检查，截断标为 `collector_limit`，不会冒充环境超时或成功。Ctrl+C / SIGTERM 在完整控制步后保存当前片段为 `interrupted`。输出目录必须为空，不覆盖旧数据。

## 数据

在已有的纯成功数据集里额外采集 1500 个成功回合（CPU 并行，可与训练同时运行）：

```bash
python -m expert.success_collection --output expert/datasets/q1_bounded_v1 \
  --successes 1500 --workers 8
```

`--successes` 是新增成功数量；失败尝试不保存，也不占用名额。

每个回合一个 `.npz` 和一个 `.json`，`index.jsonl` 汇总全部回合，`manifest.json` 记录权重、配置和代码哈希。物理状态使用 float64，观测和策略动作使用 float32。

- `qpos/qvel/qacc_warmstart/time`：T+1 个状态，保留重置前末状态。
- `observations`：T×79；`initial_history`：50×79，最新帧在前。后续历史由实际观测因果重建。
- `actions/raw_actions/pd_targets`：实际执行的归一化动作、未裁剪网络输出和 PD 目标。
- `applied_torques`：T×10×22；`foot_ball_contact`：T×10，逐物理步的真实接触。
- `ball_observation_queue`：原专家的检测/延迟队列；actor 不额外获得球真值。
- `terminated/truncated/episode_end`：区分任务终止与时间截断，不跨重置拼接。
- `world_model_valid`：所有有效物理转移，包括超时与失败。
- `policy_imitation_valid`：仅严格验证成功的完整回合为真。未成功回合仍保留，不计入完整任务模仿。

成功校验要求进球、真实脚球接触、触球后向门推进至少 0.1m、触球后存活至少 1 秒。`task_success` 同时保留专家原始进球判定，`verified_success` 为上述严格判定。超时不是摔倒，也不是成功；失败回合中的局部有效踢球可另有 `effective_kick` 标记，但不会变成整回合模仿成功。

起点固定配额 25% / 25% / 50%，通过选择原回放器的普通随机种子实现。训练、验证、测试按整回合种子在观察结果前划分，不按相邻帧拆分。`schema_check/` 和 `pilot_v1/` 为独立检查批次，原始专家数据位于 `datasets/q1_v1/`；当前训练使用带目标角限幅的 `datasets/q1_bounded_v1/`。

## 验证记录

`validation/parity.json`：8 个种子、三类起点，与原工程初态、观测、50 帧历史、动作、逐物理步状态和力矩逐位一致。

`datasets/*/dataset_check.json`、`physics_replay_check.json`、`teacher_action_check.json`：文件、标签、物理回放和教师动作校验结果。

权重、采集数据及验证输出保存在本机，不提交到 Git。此工具不执行任何策略或世界模型训练。


## 学生控制接口

原始 `runtime.py` 保留专家原行为，用于复现对照；`control.py` 的 `BoundedSoccerPlayer` 在此基础上裁剪关节目标角，最终力矩仍按模型额定范围限幅。训练、在线采集、评估和学生回放统一使用这个版本。

采集时加 `--bounded-targets`。当前带限幅数据 `datasets/q1_bounded_v1/` 共 256 回合、158411 个控制步，219 次严格成功、19 次超时、14 次出界、4 次摔倒。原数据中约 1.97% 的目标角越界，因此训练入口会拒绝直接混用旧版数据。

训练先选择 `split=train`，再过滤严格成功回合；World Model、Critic、模仿与参考全部只使用成功数据。在线未成功回合不进入 replay，也不写入新运行的在线训练归档。原始采集记录仅作历史存档，不参与训练。验证和测试划分不用于拟合。传感器缓存按原生物理过程恢复，保持 MuJoCo 最后一个子步的 IMU/相机时序。
