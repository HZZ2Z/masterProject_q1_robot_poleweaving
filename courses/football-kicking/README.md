# 足球踢球

本课程提供 Q1 视觉足球策略的训练基线。学生可以复现视觉观测、专家动作模仿和在线策略更新流程，再替换或改进算法。

训练配置和课程入口位于本目录；SuperMimic 算法适配、策略网络及训练实现位于 `shared/supermimic/q1_soccer/`，专家控制器和模型资源位于 `shared/supermimic/expert/`。演示可以直接使用本课程冒烟训练生成的 checkpoint，无需先完成正式训练。

## 环境与资产

按仓库根目录说明创建统一环境。课程正式训练使用登记的完整成功轨迹数据集；教师权重及完整数据集不纳入 Git：

- 专家权重：`expert/checkpoints/model_17800.pth`
- 训练数据：`expert/datasets/q1_bounded_v1/`
- 留出验证集：位于上述数据集目录
- 课程训练输出：`logs/auto/` 下自动生成的运行目录

若本地没有教师权重或正式数据集，请按 `shared/supermimic/expert/README.md` 联系维护者获取。只有教师权重时，也可以按下方命令现场采集小型冒烟数据集。

## 训练

先检查配置、数据和教师模型是否匹配，再开始训练：

```bash
python -m q1_soccer.train --config courses/football-kicking/configs/soccer.json --check
python -m q1_soccer.train --config courses/football-kicking/configs/soccer.json
```

短程冒烟训练使用 4 个完整教师回合，产生独立的小型数据集，不需要先准备正式训练集。输出目录须为空；如需重跑，请换一个目录，并同步修改 `configs/smoke.json` 中的 `dataset` 路径。短程训练用于验证管线并生成可供演示的 `last.pt`，不代表策略已经达到可用性能：

```bash
python -m expert.collect --bounded-targets --episodes 4 --seed 100008 \
  --threads 1 --checkpoint expert/checkpoints/model_17800.pth \
  --output logs/football-kicking-smoke-data
python -m q1_soccer.train --config courses/football-kicking/configs/smoke.json
```

输出包含 `best.pt`、`last.pt`、验证指标和运行配置，位于 `logs/q1-robot-supermimic-<时间戳>/`；`logs/latest/last.pt` 指向最近一次运行。基线记录见 [baseline.json](baseline.json)。正式训练配置为 `configs/soccer.json`。

用训练自带的 wiring、checkpoint 往返及短 rollout 检查做本地评估：

```bash
python -m q1_soccer.verify --config courses/football-kicking/configs/smoke.json \
  --device cpu --smoke --report logs/football-kicking-evaluation.json
```

## 演示

使用冒烟训练产生的 checkpoint 即可检查模型加载和无头播放；也可以换成正式训练或维护者提供的权重：

```bash
python -m q1_soccer.play --checkpoint logs/latest/last.pt --headless --steps 1000
```

演示结束后，检查命令输出中的 checkpoint 路径及步数。默认在线演示可省略 `--headless --steps`。
