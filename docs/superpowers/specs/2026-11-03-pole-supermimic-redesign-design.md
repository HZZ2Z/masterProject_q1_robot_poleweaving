# 在项目 SuperMimic 核心上实现绕杆训练

## 目标

将 `pole_task/training.py` 中独立实现的 learner，以及 `pole_task/models.py` 中自建的动力学模型和 critic，改为围绕本项目 vendored SuperMimic 核心的任务适配层。目标是在本项目中复现 task 7.7 的 SuperMimic 训练方法，以本项目 Q1 expert checkpoint 训练新的绕杆策略，并保持现有踢球训练和回放独立。

task 7.7 的训练命令通过 `python -m supermimic.train`，结合课程配置与课程目录完成训练。当前 SuperMimic 源码提交 `e023c3147e5c20af17f329e0b467538f627a27cb` 已包含课程条件和绕杆任务支持；本项目 vendored 子模块固定在较早的 `118338b19dcd14f768d1e7de5a48d04b78987688`。因此本方案保持共享 vendor 版本不变，在 pole 专属适配器中接入其公开核心 API，并移植所需的课程条件行为；不直接把新源码整体升级进共享 vendor，也不改踢球训练器。

## 目标范围

- 绕杆训练使用 vendor 的 `SuperMimic`、`WorldModel`、`Critic`、replay/update 原语、优化器和梯度处理；训练期间冻结 expert 的共享历史 encoder/action head，仅更新 zero-start pole goal encoder 与 action residual。
- 保留并加载项目记录的 `expert/checkpoints/model_17800.pth` actor 权重，作为 pole actor 的基础策略；preflight 通过 `expert/SOURCE.json` 校验 SHA-256；不加载 task 7.7 最终 actor 或 soccer student checkpoint。
- 把 expert 的 79 维、50 帧输入契约封装在 pole 专属策略/采集适配器中。参考轨迹和绕杆特征使用零初始化残差支路，使训练前动作与 expert 一致。
- 在采集、replay、想象更新和评估中始终保留参考轨迹与杆位布局配对身份；retention 片段使用显式无杆语义，避免假杆坐标进入策略、critic、world model 与 clearance 奖励。
- 真实控制与 SuperMimic 想象都在策略出口将动作裁剪到 MuJoCo 使用的 `[-1, 1]` 范围，保证 world model 和 reward 接收实际执行的动作。
- 绕杆场景保留真实物理杆；足球可见但静止且不与 pole 策略逻辑交互。
- 保留 `q1_soccer` 的导入、配置、checkpoint、训练和回放行为。

## 非目标

- 本变更不升级共享 SuperMimic 子模块提交。
- 不修改 `SoccerRobot`、`SoccerSuperMimic`、踢球配置或踢球 checkpoint 格式。
- 不把 task 7.7 的 actor 权重作为训练初始化或项目产物。
- 不把足球或球门纳入绕杆策略输入、奖励、reset 或终止逻辑。

## 方案比较

1. **在固定 vendor 核心上实现 pole 专属适配器（采用）。** 新增 pole policy、robot/course、采集适配和 `SuperMimic` 任务适配；复用 vendor 动力学拟合、replay、critic 拟合、冻结模型想象和 actor 优化，仅在任务 hook 中实现课程条件。该方案不改踢球依赖，并可保留 expert 原策略结构；代价是为 expert 的 79 维、50 帧输入增加 pole 专属 runner/history 适配，因为 vendor 默认是 72 维、10 帧。
2. **升级共享 vendor 到 task 7.7 所用的源码提交。** 可直接获得原生课程参数和绕杆支持，但 pole 与踢球会同时切换依赖，需要更新踢球的 vendor 身份锁并验证共享升级的兼容性。
3. **保留独立绕杆训练循环。** 实现速度较快，但会继续重复模型、critic 和策略优化逻辑，不符合迁移到 SuperMimic 的目标，故不采用。

## 结构设计

### PolePolicy 与 expert 初始化

`PolePolicy` 实现 SuperMimic actor 所需的调用接口，同时封装项目 `expert.policy.ActorCritic`。基础支路接收与 expert 相同的 79 维 observation 和 50 帧顺序。pole 专属适配器配置并维护这段历史；三个有序杆位在机器人坐标系中的六个数值保存在 expert 79 维切片之外。

课程/参考支路读取未来 10 帧参考窗口和当前杆位特征，末层零初始化，作为 expert 动作上的残差。构造策略时用项目 checkpoint 检查训练前动作 parity；run identity 记录 checkpoint 哈希、权重映射、输入维度和 parity 容差。task 7.7 actor 不复制、不加载，也不是必要依赖。

### PoleRobot、课程数据与 replay 身份

`PoleRobot` 将 Q1 robot/control 契约适配到派生足球展示场景，并提供绕杆 observation、课程条件、可微净空、奖励与终止特征。足球和球门对应 observation 槽保持为零。`PoleMotions` 或等价的窄适配器成对返回参考与课程数据，并保留参考 ID 和杆位顺序。

pole runner 可以管理采集节奏和场景选择，但时间序列 replay 和学习更新必须使用 vendor 数据结构及 SuperMimic 方法。若固定版本 API 缺少 pole 专属操作，由 `PoleSuperMimic` 仅覆盖任务 hook；模型拟合、critic 拟合、梯度裁剪和 optimizer 更新仍沿用 vendor 实现。

### 模型和训练流程

`PoleWorldModel`、`PoleCritic` 应为 vendor 模型的薄子类/配置，或直接使用 vendor 实例；不得保留独立 MLP 架构和独立 optimizer loop。课程身份通过 pole context 进入动力学模型；模型用真实 transition 训练，终止标签同时区分跌倒和真实杆接触。

actor 更新沿用 SuperMimic 的 imagination 与 critic 流程，pole 适配器只提供路线奖励、净空/接触惩罚和终止 hook。参考窗口长度为 10 帧，包含机器人根部相对位置/朝向和关节参考；杆位特征为三个按路线排序的杆中心，在机器人朝向坐标系中表示。参考与杆位共享相同的场景平移变换。

actor 更新前要求 held-out transition RMSE <= 0.15、termination BCE <= 0.10、真实回报 critic 相对 RMSE <= 0.35。actor 学习率使用成功的 task 7.7 route-adapter 配置 `3e-4`，并在 actor 出口裁剪动作，使 imagined 与 MuJoCo 实际动作一致。retention 片段屏蔽虚假的杆坐标和 next-pole 条件，但保留横向/朝向跟踪误差供 route adapter 学习基础步态保持。每个 accepted stage 相对上一接受策略的信任门限为动作变化 RMS <= 0.2、最大绝对变化 <= 0.5，与 adapter 的 `0.5` 残差尺度一致。每 25 次 actor update 做完整真实回放；路线 RMSE 相对 expert baseline 回退超过 0.1 m，或 retention 回退时恢复最后接受的策略并停止 actor 更新。训练最多 2500 次 actor update，允许最多 100 次评估耐心；最终 checkpoint 选择仍按严格验收阈值。上述阈值均可由 `pole_task/config.json` 配置。

本地足球专家在 straight/right-turn 参考上的基线 RMSE 高于 0.3 m，因此三段 retention 参考也进入训练回放，供零初始化 route adapter 学习项目内的参考跟踪修正；held-out S 路线仍不进入训练。

奖励沿用 SuperMimic 的参考追踪与跌倒语义，并增加注册的 pole-clearance 项；实际机器人杆接触立即标记本回合失败和终止。评估沿用 OpenSpec 已登记标准：nominal 675 步与 held-out 671 步路线须完成、终点距离 <= 0.5 m、无碰杆/跌倒且 root XY RMSE <= 0.5 m；straight/left/right retention clip 无跌倒且 RMSE <= 0.3 m。精确奖励实现应对照来源 task 7.7 的有效训练配置，不从当前独立 learner 的临时系数推导。

### 与踢球管线的兼容边界

保持 `3rdparty/SuperMimic` 固定提交以及 `q1_soccer/vendor.py` 身份锁不变。pole task 使用独立的模块、history/采集包装和运行入口；既有踢球路径继续使用它自己的 79x50 vision policy、109 维球状态、配置和 checkpoint。需要扩展 vendor 通用 API 时，必须作为可选参数或 pole 专属适配实现，不能改变默认路径。

### 验收和回放

无显示器环境下运行 headless pole 评估，覆盖 nominal 675 步、held-out 671 步和 straight/left/right retention clip。报告路线完成、终点距离、杆碰撞、跌倒、root XY RMSE 和综合成功标志。GUI 回放保留为用户后续视觉检查入口，不在当前实现过程中打开。

## 风险与处理

- **vendor runner 将历史长度写死为 10。** 在 pole 专属采集/replay 路径中保留 expert 的 50 帧历史，并确认算法将 history 作为输入数据处理；不改踢球默认值。
- **课程信息在 replay 或想象中丢失。** 每条 replay transition 存课程上下文和参考 ID，并在 reset、采集、batch 采样和评估时核对配对身份。
- **杆碰撞是模拟器物理事件。** 真实接触标签由物理仿真记录；想象中使用可微净空近似，真实接触始终终止回合。
- **expert 与 vendor 策略输入形状不同。** 保留 expert actor 作为策略基础，只添加零初始化的参考/杆位残差路径；checkpoint 不兼容时在创建 run 目录前报错。
- **踢球行为可能受共享修改影响。** 不改 vendor pin 和 soccer 模块，并保留独立的 soccer smoke 验收任务。

## 实施顺序

1. 将 model 层改为 vendor-compatible expert wrapper 与 pole 任务薄适配器，移除独立 world model/critic 实现。
2. 改写训练编排，调用 SuperMimic 核心模型拟合、replay、critic 和 actor 更新；只保留 pole 专属采集、验证门禁、信任监控和 run manifest。
3. 使 OpenSpec 任务清单与该边界一致，再完成课程模拟、评估与验收，不打开 UI。

## 留给实现者的选择

- 若 vendor 方法已有合适 hook，优先决定具体适配类边界而不复制算法代码。
- pole collector 可继承 vendor collector，或用轻量 50 帧历史 wrapper 组合 vendor collector。
- 课程 context 的设备编码方式可由实现者确定，但必须保证所有训练路径的身份一致。
