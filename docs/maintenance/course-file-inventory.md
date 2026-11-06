# 课程文件归属清单

本清单对应 `lecture` 上已整合的三个来源分支。目标路径按学生任务和实际代码依赖划分；迁移时应保持来源历史可追溯，并更新所有依赖路径和测试入口。

## 课程与框架文件

| 当前路径或来源 | 目标位置 | 归属与迁移说明 |
| --- | --- | --- |
| `q1_soccer/`、`expert/` | `shared/supermimic/` | SuperMimic 的训练框架和专家策略同时被踢球与绕杆流程引用；作为两门课程共用的框架维护。保留专家数据、Q1 场景、动作和来源清单。 |
| `3rdparty/SuperMimic/` | 仓库根目录 `3rdparty/SuperMimic/` | 保留上游子模块路径，避免未来同步来源分支时再次迁移 gitlink。 |
| `.gitmodules` | 仓库根目录 | 保留 `3rdparty/SuperMimic` 的子模块路径，避免后续同步上游时重复迁移 gitlink。 |
| `configs/soccer.json`、`baselines/q1-robot-supermimic-baseline-*.json` | `courses/football-kicking/` | 踢球课程的配置和基线记录。 |
| `pole_task/`、`q1_supermimic/`、`docs/pole-weaving.md` | `courses/pole-weaving/` | 绕杆任务、配置、评估、回放和课程说明。`pole_task` 依赖共享 SuperMimic 框架与专家模型。 |
| `q1_4v4/`、`train.py`、`play.py`、`train_distributed.py`、`CURRICULUM.md` | `courses/soccer-4v4/` | 4v4 课程的环境、训练、回放和课程阶段说明。仓库根的 `train.py`、`play.py` 和 `train_distributed.py` 保留兼容链接。 |
| `third_party/mappo/` | `courses/soccer-4v4/third_party/mappo/` | 4v4 使用的 MAPPO 上游代码，保持其许可证和来源说明；根 `requirements.txt` 使用此路径安装。 |
| `resources/Q1.yaml`、`resources/motions/q1/`、`resources/q1_description/q1_4v4.xml`、`artifacts/scene_4v4.png` | `courses/soccer-4v4/resources/` 或该课程的 `artifacts/` | 4v4 配置、运动先验、场景和可视化资源。专属 4v4 场景迁入课程目录；机器人网格与单机器人场景使用共享资源。 |
| `scripts/audit_marching_*.py`、`scripts/create_dashboard.py`、`scripts/python.sh`、`scripts/smoke*.py`、`scripts/watch_training.py` | `courses/soccer-4v4/scripts/` | 这些脚本导入 `q1_4v4`，与课程代码一起迁移并通过物理脚本路径定位仓库根目录；仓库根保留兼容链接。 |

## 共享与兼容内容

| 当前路径 | 目标位置 | 归属与迁移说明 |
| --- | --- | --- |
| `envs/`、`utils/`、`data/`、`resources/` 中未归入课程的资产 | `shared/q1-mujoco/` | 保留当前 Q1/G1/T1 MuJoCo 环境、工具和动作数据，独立于三个课程的训练算法。通过兼容入口维持现有导入和测试路径。 |
| `play_mujoco.py`、`evaluate_q1.py`、`scripts/prepare_q1.py`、`scripts/retarget_q1_motions.py` | `shared/q1-mujoco/` | 当前通用仿真回放、评估和资源准备工具。 |
| `expert/resources/q1_description/` 与 `resources/q1_description/` 中同哈希的 Q1 模型文件 | `shared/q1-mujoco/resources/q1_description/` | `q1_soccer.xml` 和机器人网格已确认相同；该目录作为规范资源位置，绕杆专属源任务文件也放在此目录。`shared/supermimic/expert/resources` 通过符号链接兼容原有引用。 |
| `tests/` | 仓库根目录 `tests/` | 保持单一 pytest 入口；兼容链接维持原导入路径，测试按模块名区分通用环境、踢球、绕杆和 4v4 行为。 |

迁移后保留顶层兼容符号链接：`q1_soccer/` 和 `expert/` 指向 `shared/supermimic/`；`pole_task/` 和 `q1_supermimic/` 指向 `courses/pole-weaving/`；`q1_4v4/` 指向 `courses/soccer-4v4/`；`envs/`、`utils/`、`data/`、`resources/`、`play_mujoco.py` 和 `evaluate_q1.py` 指向 `shared/q1-mujoco/`。`train.py` 和 `play.py` 保留轻量兼容入口，实际 4v4 实现放在课程目录。共享目录中的 `expert/resources` 还通过符号链接复用规范 Q1 资源。

## 入口和环境文件

| 当前路径 | 目标位置 | 迁移说明 |
| --- | --- | --- |
| 根 `environment.yml`、`requirements.txt`、`Dockerfile`、`.dockerignore` | 仓库根目录 | 保留一个包含三门课程依赖的统一环境入口，更新 MAPPO editable 安装位置并明确子模块初始化步骤。Docker 保持项目级构建。 |
| `expert/requirements.txt` | `shared/supermimic/` | 作为踢球和绕杆共用框架的依赖补充，由两门课程说明引用。 |
| 根 `README.md` | 仓库根目录 | 只保留项目介绍、通用安装提示、两类上手入口和三门课程导航。 |
| `tests/` | 仓库根目录 | 保持从仓库根目录运行测试的入口，避免测试路径迁移与课程代码路径迁移同时发生。 |

根目录保留 `train.py` 和 `play.py` 作为 4v4 兼容包装脚本；课程级实现放在 `courses/soccer-4v4/`。根 `pytest.ini` 继续提供项目级测试设置。

## 已确认的依赖关系

- 绕杆的 `pole_task` 直接使用 `expert.policy`，并通过 SuperMimic profile 进入 `q1_soccer` 训练框架，因此这两套 SuperMimic 包应共享同一份框架代码。
- 4v4 的 `q1_4v4`、训练脚本和测试导入路径彼此耦合，但不依赖通用 `envs/` 或 `utils/`；移动时需同步迁移课程脚本、资源和路径测试。
- 当前通用 `envs/` 与 `utils/` 被 `play_mujoco.py`、Q1/G1 测试使用，应整体保留为共享兼容层，不并入 MAPPO 或 SuperMimic 算法包。
- `expert/resources/q1_description/q1_soccer.xml` 与 `resources/q1_description/q1_soccer.xml` 的 SHA-256 相同；`meshes/pelvis.STL` 也相同。重复的根资源在迁移时可去重，但来源清单和绕杆专属源任务文件必须继续可用。
- `data/left_kick_1.csv` 与 `resources/motions/q1/left_kick_1.csv` 内容不同，分别保留在通用数据和 4v4 课程资源中。
