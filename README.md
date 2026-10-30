# Q1 Humanoid Kick / SuperMimic

Q1 V3（22 自由度）的视觉踢球训练。GPU 使用 MuJoCo Warp，CPU 原生 MuJoCo Batch 用于独立验收和 GUI。与相邻 PPO 项目共用 Conda 依赖，但环境代码、资源、进程、权重和日志独立。当前尚未证明踢球收敛。验收必须包含踢球课程中的真实进展：有效踢球、向球门净推进及成功率改善；站立和模型检查仅是中间检查。

## 当前状态：正式训练停止，主线修复中

2026-10-30 的正式 run 在第 2300 轮保存停止。最后一次 actor 更新在第 489 轮，第 500 与 2300 轮的 actor 权重完全相同，足球课程没有开启。此前单元测试与短测通过不构成训练有效的证据，完整数据见 `VALIDATION.json` 的 `runtime_audit`。

当前主线已修正物理模型输入和增量归一化、critic 状态表示、课程统计与真实策略验收。三组独立场景的短程候选更新均改善了带噪存活率和回报，但尚未证明持续学习或踢球收敛。连续 200 轮没有实际或保留策略更新会保存停止；单元测试通过不作为训练成功的依据。

## 结构和上游

- `envs/`、`resources/`、`data/`、`utils/`：Q1 物理、视觉观测、奖励、资源和参考动作的独立副本。
- `q1_supermimic/environment.py`：终止前状态采集；`learning.py`：世界模型和真实轨迹学习；`safety.py`：真实物理验收；`metrics.py`：回合统计。
- `train.py`、`play.py`：训练和回放；`configs/soccer.yaml`：训练参数；`envs/Q1.yaml`：任务参数。
- `3rdparty/SuperMimic`：Git submodule，固定 `118338b19dcd14f768d1e7de5a48d04b78987688`。克隆本项目后运行 `git submodule update --init`。
- `SOURCE.json` 记录来源，`VALIDATION.json` 记录验证结果。运行数据全部在忽略的 `logs/` 下。

上游 [SuperMimic](https://github.com/PrimebotTech/SuperMimic) 是 Q1 V2 动作跟踪。本项目是足球适配，不直接调用其完整训练器。目前复用上游 MLP 构建组件、自回归世界模型监督思路；不是论文完整复现。

## 当前训练实现

只维护这一套训练主线。物理模型与带噪视觉/历史观测分离，读取物理与执行器状态、质量、摩擦、完整重心和惯量，按单步速度增量学习，再积分位置和姿态。模型宽度为 512，actor/critic 为 256。根加速度奖励统一使用世界坐标。

课程存活率按环境等权汇总每个环境最近一次已完成的 5 秒试验，避免快速失败的环境因反复重置而获得额外权重，也避免早期失败在策略改善后长时间拖住课程；回合结束指标继续按回合统计，二者含义不同。单步奖励响应仍不可靠，因此必须以多步真实物理中的策略改善验收；当前不能视为可用踢球成果。

策略读取 79 维虚拟视觉/本体观测及 4 帧观测历史，不读取特权状态。每轮真实采样 64 个控制步；动作范围为完整的 ±1，初始均值为零，探索标准差为 0.025。该尺度经固定种子的噪声对照选择，不直接照搬 PPO 的探索尺度。世界模型的动作输入按 warmup 实测标准差归一化，归一化不限制真实动作幅度。

恢复模型驱动的短视野解析策略梯度：冻结世界模型和 critic 参数，策略经过 8 步预测，优化已知平滑物理/足球目标与终点价值。想象中的动作噪声与真实采样一致，回溯步长时复用同一组噪声；高斯 KL 上限为 0.01，同时约束相对上一轮策略和最近真实验收策略的累计变化，避免多轮小步累计成未经验证的大步。学习奖励头和终止头不参与 actor 目标。离散进球、触球奖励保留在真实 critic 目标和物理验收中，平滑目标不宣称与所有接触奖励逐项等价。

Critic 读取紧凑物理状态、相对位置、课程上下文与物理参数，排除视觉噪声、观测历史和无关的绝对场地位置。使用当前真实轨迹的固定 GAE 目标（gamma=0.995、lambda=0.98），默认拟合 4 个 epoch；成功/失败终止不自举，人工时限截断使用重置前观察自举，优势不跨自动重置。与上游的区别是 critic 目标以真实转移为依据，而不是完全使用想象回报；这仍是足球适配，不是完整上游跟踪训练器。

世界模型预测速度变化，使用梯形积分和局部角速度四元数更新重建位置、姿态，再从物理状态生成本体/特权观测并推进视觉延迟队列与历史。短视野内保持可见性和测量残差不变，这是近似，不是完整相机模型。足球计时和接近窗口按真实规则逐步推进；脚球距离通过同一 URDF 的正运动学和碰撞包围盒计算，停滞奖励读取更新后的状态。

世界模型进行 8 步自回归监督，1/8 环境留出验证。除了整段物理误差，还在独立 MuJoCo 成对场景施加 ±动作扰动，检查响应方向（余弦至少 0.8）和相对平方误差（最多 0.4）。躯干位姿和单步奖励响应保留为诊断，最终候选更新需要通过多步真实物理比较。通过门槛后，最早第 50 轮允许策略更新。`learning/model_policy_active` 显示是否实际应用优化器更新；未通过时只训练模型和 critic，不暗中切回 VPG。

## 任务、课程和显示

1024 环境固定为 256 / 256 / 512，局部重置保持组别：

| 比例 | 起点 |
|---|---|
| 25% | 门线以内：x∈[-6.5,6.5]、y∈[-4,4]，拒绝初始重叠 |
| 25% | 半径 3.5m 圆内按面积均匀采样，拒绝初始重叠 |
| 50% | 朝向球、中立安全姿态、初始速度零，最近脚球间隙 10–25cm |

最早第 100 轮且存活率达标后，近球组先启用完整足球奖励；停滞时削减站立收入、加强接近球信号，练习回合 12 秒。其他两组先站立；近球成功率达到门槛后再开放全场任务。参考动作初始化和外力扰动随全场课程推进。当前动作上限独立于课程，随权重保存。

场景、草皮、球、场线和回放相机与 PPO 对齐，F 键切换视锥。基础足球奖励项与来源项目对齐，但本项目没有 AMP，且课程权重不同，所以总奖励不能直接等同于 PPO。

## 环境、检查和训练

使用现有 `/home/kemove/miniconda3/envs/q1-kick` 环境，不安装上游整套 pyproject 依赖。依赖见 `requirements.txt`；资源许可证保留在原目录。

```bash
./scripts/check.sh
./scripts/run.sh --preflight --device cuda:0
./scripts/run.sh --smoke --device cuda:0
./scripts/run.sh --start-training --iterations 20000  # 全新训练，不传 checkpoint
```

`--preflight` 仅 4 个控制步；`--smoke` 固定两轮且不上报 W&B；`--start-training` 才是正式训练。`iterations` 为累计目标。SIGINT/SIGTERM 会在当前迭代结束后保存。当前训练位置见 `logs/active_training.json`。

状态和网络结构不兼容的旧 checkpoint 不可加载。新训练从随机初始化开始，不传 `--checkpoint`；需要续训时，仅传与当前任务、网络结构相容的 checkpoint。恢复网络、优化器、归一化、课程和 RNG，重建物理轨迹、replay 与回合统计窗口。

W&B 项目为 `mohismlab-robotics/q1-humanoid-kick-SuperMimic`。凭据读取已有登录或环境变量，禁止提交到 Git。Torch allocator 限总显存 30%，Warp 另计；1024 环境用于给并行 PPO 留出显存。

## 验收、best 和 GUI

每 10 轮使用三组各 64 个 CPU MuJoCo 环境运行 6 秒首回合；每次更换种子，候选策略与最近接受策略使用完全相同的场景及探索噪声。平衡阶段同时检查确定性和带噪回报不明显退化，并以 Wilson 下界容忍有限样本的存活率波动。足球阶段要求至少 65% 存活、不低于中立基准、失败率不超过 35%。这些检查不是独立统计显著性证明。拒绝候选时只恢复 actor、对应优化器和动作上限，保留世界模型与 critic 从真实失败中学到的内容。未变化的策略不计为新接受更新。

足球 best 独立选优：必须满足安全底线，出现有效踢球事件，且近球组平均净向球门推进至少 1cm。踢球事件按脚球间隙、脚速、朝门球速、直立且未摔倒判断，是任务事件判据，不是独立触觉传感器。净推进使用初末球门距离差，避免累加来回抖动。

```bash
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
    -u LD_PRELOAD -u PYOPENGL_PLATFORM -u WAYLAND_DISPLAY \
    MUJOCO_GL=glfw XDG_SESSION_TYPE=x11 \
    /home/kemove/miniconda3/envs/q1-kick/bin/python play.py \
    --checkpoint <当前运行目录>/last.pt --accepted --viewer --steps 1000000000
```

`--accepted` 显示最近安全策略；`--best` 只加载已有踢球证据的足球 best，不存在时明确报错。都恢复对应动作上限。GUI 启动时加载权重，训练更新后需重新启动。

## 看哪些指标

- 稳定性：`stability/reset_cohort_survival_5s`；真实评估看 `evaluation/survival_5s`。
- 踢球能力：`evaluation/near_kick_event_rate`、`evaluation/near_net_goal_progress_m`、`success_rate`。
- 是否取得足球 best：`evaluation/soccer_best_available`。
- `fall_rate`、`reward`、`steps`、`episode/*` 为最近 1000 个已结束回合的加权均值。结合 `statistics/window_episodes` 与 `window_ready`；`instant/*` 仅排查瞬时异常。结束回合统计仍有选择偏差，不能替代重置队列存活率。
- `model_validation/*` 包括预测和动作响应验证；通过只表示允许尝试模型更新，不代表会踢球。
- `evaluation/exploratory_survival_5s` 检查带训练噪声的稳定性，不能用确定性存活率代替。

## 审计与方法资料

代码修改使用 Git 记录。只有当前主线，不将每次实验命名为新产品版本。

短测发现确定性策略验收可能掩盖探索噪声下的退化，因此同时验证确定性和带噪策略。验证使用独立且轮换的种子场景，正式训练重新随机初始化，不加载验证权重。具体结果见 `VALIDATION.json`，不能把单元测试或短测通过当成足球收敛。

参考：[上游算法实现](https://github.com/PrimebotTech/SuperMimic/blob/main/src/supermimic/algorithm.py)、[GAE 原论文](https://arxiv.org/abs/1506.02438)、[Gymnasium 终止与截断](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。
