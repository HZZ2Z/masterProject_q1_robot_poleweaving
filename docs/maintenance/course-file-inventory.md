# 课程文件归属与入口迁移

课程实现只在规范目录维护。先在仓库根目录按[环境指南](../setup.md)执行 `python -m pip install --no-deps -e .`，之后 Python 包从下表对应源码位置导入。仓库根目录不再提供兼容软链接。

## 课程专属内容

| 规范位置 | 内容 | Python 包 / 命令 |
| --- | --- | --- |
| `courses/football-kicking/` | 踢球课程配置、说明和基线 | `q1_soccer` 包由 `shared/supermimic/q1_soccer/` 提供；配置位于本目录。 |
| `courses/pole-weaving/pole_task/` | 绕杆环境、训练、评估、回放与课程数据 | `pole_task` 包；命令使用 `python -m pole_task.<模块>`，配置路径为 `courses/pole-weaving/pole_task/config.json`。 |
| `courses/pole-weaving/q1_supermimic/` | 绕杆的 SuperMimic 训练扩展 | `q1_supermimic` 包。 |

## 跨课程共享内容

| 规范位置 | 内容 | Python 包 / 资源 |
| --- | --- | --- |
| `shared/supermimic/q1_soccer/` | 踢球与绕杆共用的训练、模型和策略代码 | `q1_soccer` |
| `shared/supermimic/expert/` | 专家策略、数据、配置与工具 | `expert` |
| `shared/q1-mujoco/envs/` | 通用 MuJoCo 环境 | `envs` |
| `shared/q1-mujoco/utils/` | 仿真、动作和训练工具 | `utils` |
| `shared/q1-mujoco/` | 通用仿真脚本、模型、动作数据和资源 | `play_mujoco`、`evaluate_q1`；资源位于 `shared/q1-mujoco/resources/`。 |

可编辑安装通过项目 `setup.py` 映射上述 Python 包到唯一源码目录。仓库级 `tests/conftest.py` 只为测试进程加入规范源码路径，不提供运行时兼容入口。

## 旧入口迁移表

| 已移除的根入口 | 新位置或方式 |
| --- | --- |
| `data`、`envs`、`utils`、`resources` | 分别使用 `shared/q1-mujoco/data/`、`shared/q1-mujoco/envs/`、`shared/q1-mujoco/utils/`、`shared/q1-mujoco/resources/`。 |
| `expert` | `shared/supermimic/expert/` |
| `q1_soccer` | `shared/supermimic/q1_soccer/`（安装后仍以 Python 包名 `q1_soccer` 导入）。 |
| `pole_task`、`q1_supermimic` | `courses/pole-weaving/pole_task/`、`courses/pole-weaving/q1_supermimic/`。 |
| 根目录 `play_mujoco.py`、`evaluate_q1.py` | `shared/q1-mujoco/play_mujoco.py`、`shared/q1-mujoco/evaluate_q1.py`。 |

旧模块导入应先执行可编辑安装；课程包不再从根目录复制或链接源码。

## 维护约定

- 修改课程实现时只改对应 `courses/<课程>/` 内的规范文件；修改共享实现时只改 `shared/` 内的规范文件。
- 新增课程或共享包时，在 `setup.py` 映射到规范源码位置，并同步维护本文。
- 课程训练数据和模型产物存放在 `logs/` 或课程指定目录，不纳入 Git。
- 通用数据和课程专属数据即使文件名相同，也依据用途和内容判断归属，不因名字相同而合并。
