# Q1 机器人足球课程

本项目提供三门围绕启元 Q1 人形机器人的实践课程。每门课程都包含一个可运行的算法基线；学生可以复现实验，并在课程目录中独立替换或迭代算法。

## 课程

- [足球踢球](courses/football-kicking/README.md)：视觉足球策略、专家轨迹与策略训练。
- [S 型绕杆](courses/pole-weaving/README.md)：沿三杆路线完成连续绕行。
- [4v4 足球对抗](courses/soccer-4v4/README.md)：从基础步态逐步训练到团队比赛。

## 快速开始

需要 Linux、Conda/Mamba，以及支持 CUDA 12.8 的 NVIDIA GPU。创建统一环境：

```bash
git submodule update --init --recursive
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple conda env create -f environment.yml
conda activate q1-kick-4v4
```

环境文件通过 pip 安装 Python 依赖；PyTorch CUDA wheel 另从 `requirements.txt` 声明的官方 PyTorch 索引获取。

查看 4v4 基础场景的无头演示（不加载模型）：

```bash
python play.py --neutral --headless --steps 30 --device cuda:0
```

按所选课程准备训练数据或模型，再运行课程指南中的训练和策略演示命令。演示和训练输出写入本地 `logs/`，模型和训练数据不会提交到 Git。

## 实验复现

选择一门课程并按对应快速开始说明执行。课程页面列出依赖、基线算法、训练配置、模型路径、评估方式和演示命令。根目录保留旧脚本导入路径，新的开发和实验入口以 `courses/` 下的任务目录为准。

通用 MuJoCo 环境、机器人模型和动作数据位于 `shared/q1-mujoco/`；SuperMimic 训练框架位于 `shared/supermimic/`。
