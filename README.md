# Q1 Humanoid Kick / SuperMimic

Q1 V3（22 自由度）的视觉踢球训练。GPU 使用 MuJoCo Warp，CPU 原生 MuJoCo Batch 用于独立验收和 GUI。与相邻 PPO 项目共用 Conda 依赖，但环境代码、资源、进程、权重和日志独立。当前尚未证明踢球收敛。

## 结构和上游

- `envs/`、`resources/`、`data/`、`utils/`：Q1 物理、视觉观测、奖励、资源和参考动作的独立副本。
- `q1_supermimic/environment.py`：终止前状态采集；`learning.py`：世界模型和真实轨迹学习；`safety.py`：真实物理验收；`metrics.py`：回合统计。
- `train.py`、`play.py`：训练和回放；`configs/soccer.yaml`：训练参数；`envs/Q1.yaml`：任务参数。
- `3rdparty/SuperMimic`：Git submodule，固定 `118338b19dcd14f768d1e7de5a48d04b78987688`。克隆本项目后运行 `git submodule update --init`。
- `SOURCE.json` 记录来源，`VALIDATION.json` 记录验证结果。运行数据全部在忽略的 `logs/` 下。

上游 [SuperMimic](https://github.com/PrimebotTech/SuperMimic) 是 Q1 V2 动作跟踪。本项目是足球适配，不直接调用其完整训练器。目前复用上游 MLP 构建组件、自回归世界模型监督思路；不是论文完整复现。

## 当前学习路径（checkpoint v4）

策略只读取前 79 维虚拟视觉/本体观测。每轮真实采样 64 个控制步，保存 tanh 前高斯样本；动作上限是策略的一部分，恢复/回滚后不会被课程重写。探索标准差固定为 0.015。

真实轨迹以 gamma=0.995、lambda=0.98 计算 GAE，优势和价值目标在更新前一次性冻结。终止（包括成功）不自举；人工时限截断使用重置前观察自举；两者均截断跨回合优势传播。Critic 只拟合当前策略的固定目标，默认 4 个 epoch。每轮一次 VPG 更新，以解析高斯 KL 上限 0.005 和代理目标改善检查回溯步长；不是 PPO，也不是完整 TRPO。恢复后先做 20 轮价值拟合。

世界模型仍以真实 replay 多步监督，并留出 1/8 环境验证。其预测误差不能证明动作梯度可信，因此审计后移除了模型梯度驱动 actor 的路径；模型当前不决定策略更新。新增学习诊断仅为 KL、实际更新比例、价值拟合前后 MSE 和 explained variance。

## 任务、课程和显示

1024 环境固定为 256 / 256 / 512，局部重置保持组别：

| 比例 | 起点 |
|---|---|
| 25% | 门线以内：x∈[-6.5,6.5]、y∈[-4,4]，拒绝初始重叠 |
| 25% | 半径 3.5m 圆内按面积均匀采样，拒绝初始重叠 |
| 50% | 朝向球、中立安全姿态、初始速度零，最近脚球间隙 10–25cm |

近球组先启用完整足球奖励；停滞时削减站立收入、加强接近球信号，练习回合 12 秒。其他两组先站立；近球成功率达到门槛后再开放全场任务。参考动作初始化和外力扰动随全场课程推进。当前动作上限独立于课程，随权重保存。

场景、草皮、球、场线和回放相机与 PPO 对齐，F 键切换视锥。基础足球奖励项沿用 v4，但本项目没有 AMP，且课程权重不同，所以总奖励不能直接等同于 PPO。

## 环境、检查和训练

使用现有 `/home/kemove/miniconda3/envs/q1-kick` 环境，不安装上游整套 pyproject 依赖。依赖见 `requirements.txt`；资源许可证保留在原目录。

```bash
./scripts/check.sh
./scripts/run.sh --preflight --device cuda:0
./scripts/run.sh --smoke --device cuda:0
./scripts/run.sh --start-training --checkpoint <v4-checkpoint.pt> --iterations 20000
```

`--preflight` 仅 4 个控制步；`--smoke` 固定两轮且不上报 W&B；`--start-training` 才是正式训练。`iterations` 为累计目标。SIGINT/SIGTERM 会在当前迭代结束后保存。当前训练位置见 `logs/active_training.json`。

v3 需要显式 `--migrate-learner`：保留已验收 actor、对应动作上限、actor 优化器、世界模型和归一化；重建曾使用旧策略 TD 的 critic。v4 正常续训恢复所有网络及优化器；物理轨迹、回放池和统计窗口重新开始。仅在任务课程变化时使用 `--migrate-curriculum`，不允许它掩盖奖励/物理/控制不兼容。

W&B 项目为 `mohismlab-robotics/q1-humanoid-kick-SuperMimic`。凭据读取已有登录或环境变量，禁止提交到 Git。Torch allocator 限总显存 30%，Warp 另计；1024 环境用于给并行 PPO 留出显存。

## 验收、best 和 GUI

每 50 轮用固定种子的 32 个 CPU MuJoCo 环境运行 6 秒首回合，指标均来自真实物理。足球阶段安全底线是至少 65% 的 5 秒存活率、不低于中立基准且失败率不超过 35%。不满足时恢复 actor、critic、两个优化器和动作上限；安全的接近距离小幅变化不会再触发回滚。

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

- 稳定性：`stability/reset_cohort_survival_5s`；固定评估看 `evaluation/survival_5s`。
- 踢球能力：`evaluation/near_kick_event_rate`、`evaluation/near_net_goal_progress_m`、`success_rate`。
- 是否取得足球 best：`evaluation/soccer_best_available`。
- `fall_rate`、`reward`、`steps`、`episode/*` 为最近 1000 个已结束回合的加权均值。结合 `statistics/window_episodes` 与 `window_ready`；`instant/*` 仅排查瞬时异常。结束回合统计仍有选择偏差，不能替代重置队列存活率。
- `model_validation/*` 仅表示模型拟合质量，不代表策略会踢球。

## 2026-10-30 审计结论与资料

v3 的问题包括：旧策略 replay 自举的 critic 被用于当前策略优势；只用单步 TD 传递延迟奖励；模型状态误差被当作策略梯度可靠性依据；动作上限回滚被下一轮覆盖；成功被当作可自举超时；几毫米接近回落导致连续回滚。v4 对应修复见上述学习与验收路径。历史实现由 Git 基线保留，避免文档并存多套过期说明。

参考：[GAE 原论文](https://arxiv.org/abs/1506.02438)、[Gymnasium 终止与截断](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)、[TRPO 原论文](https://arxiv.org/abs/1502.05477)及本地上游 `algorithm.py` 的固定 lambda-return 目标。借鉴 KL 约束并不等于实现了论文的自然梯度算法。

测试和短时实训只能验证实现语义与局部稳定性，不能证明踢球收敛；具体实测见 `VALIDATION.json`。
