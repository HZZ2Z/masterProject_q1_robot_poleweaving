# 课程文件归属清单

本文记录当前规范目录与仓库根兼容入口。课程实现只在规范目录维护；根目录符号链接用于兼容现有 Python 导入和命令，不是第二份源码。

## 课程专属内容

| 规范位置 | 内容 | 说明 |
| --- | --- | --- |
| `courses/football-kicking/` | 踢球课程说明、训练配置与基线信息 | 训练代码通过 `q1_soccer` 使用共享 SuperMimic 框架。 |
| `courses/pole-weaving/pole_task/` | 绕杆环境、训练、评估、回放、课程数据 | 绕杆任务代码唯一来源。 |
| `courses/pole-weaving/q1_supermimic/` | 绕杆使用的 SuperMimic 训练入口/扩展 | 唯一源码位于课程目录；根 `q1_supermimic` 是兼容符号链接。 |
| `courses/soccer-4v4/q1_4v4/` | 4v4 环境、训练算法、课程与比赛逻辑 | 唯一源码位于课程目录；根 `q1_4v4` 是兼容符号链接。 |
| `courses/soccer-4v4/{train.py,train_distributed.py,play.py,scripts/,resources/,third_party/}` | 4v4 入口、辅助脚本、场景资源与 MAPPO 上游依赖 | 课程专属文件均归 4v4 目录。根训练入口和脚本链接仅作兼容。 |

## 跨课程共享内容

| 规范位置 | 内容 | 根目录兼容入口 |
| --- | --- | --- |
| `shared/supermimic/q1_soccer/` | 踢球与绕杆共用的训练、模型、策略、数据和运行代码 | `q1_soccer -> shared/supermimic/q1_soccer` |
| `shared/supermimic/expert/` | 共用专家策略、数据与工具 | `expert -> shared/supermimic/expert` |
| `shared/q1-mujoco/` | 通用 MuJoCo 环境、工具、动作数据和机器人模型 | `envs`、`utils`、`data`、`resources`、`play_mujoco.py`、`evaluate_q1.py` 等链接 |

`3rdparty/SuperMimic/` 子模块和 `.gitmodules` 保持仓库根路径。根 `tests/`、环境文件、Docker 文件及项目 README 作为仓库级入口维护。

## 根目录兼容入口

以下根路径当前为符号链接，不能在根目录另存一份实现：

- `q1_soccer -> shared/supermimic/q1_soccer`
- `q1_supermimic -> courses/pole-weaving/q1_supermimic`
- `q1_4v4 -> courses/soccer-4v4/q1_4v4`
- `pole_task -> courses/pole-weaving/pole_task`
- `train.py`、`train_distributed.py`、`play.py` 及 4v4 辅助脚本指向课程内入口
- `expert`、`envs`、`utils`、`data`、`resources` 及通用仿真脚本指向 `shared/` 内规范实现

## 维护约定

- 修改课程实现时只改对应 `courses/<课程>/` 内的规范文件；修改共享实现时只改 `shared/` 内的规范文件。
- 新增或移除兼容入口时，同步更新本文并检查链接目标和现有导入。
- `q1_slalom/` 不是项目源码目录；目前只发现被忽略的 Python 缓存，且没有代码引用。缓存清理后不应重新提交或作为课程实现维护。
- 通用数据和课程专属数据即使文件名相同，也须依据用途和内容判断归属，不因名字相同而合并。
