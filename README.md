# Q1 Robot Soccer Supermimic

基于启元官方 [SuperMimic](https://github.com/PrimebotTech/SuperMimic)，接入 Q1 V3 踢球专家、视觉观测和足球场景。

## Quick Start

```bash
git clone --recurse-submodules --branch SuperMimic git@github.com:MohismLab/Q1-Robot-Soccer.git
cd Q1-Robot-Soccer
conda env create -f environment.yml
conda activate q1-supermimic
```

使用仓库记录的专家权重 `expert/checkpoints/model_17800.pth`。准备带关节目标角和力矩限幅的数据：

```bash
python -m expert.collect --bounded-targets --episodes 256 --seed 100000 \
  --output expert/datasets/q1_bounded_v1
python -m expert.check_dataset expert/datasets/q1_bounded_v1
python -m expert.verify_teacher expert/datasets/q1_bounded_v1
```

检查接线或运行短冒烟测试：

```bash
python train.py --check
python -m q1_soccer.verify --device cpu --smoke
```

数据采集与训练默认使用该专家权重；如需换教师，采集和验证命令都应显式传入同一个 `--checkpoint`，并将 `configs/soccer.json` 的 `checkpoint` 改为相同文件。足球播放器仍只加载项目训练生成的学生 checkpoint。

启动训练：

```bash
python train.py --config configs/soccer.json
```

配置位于 `configs/soccer.json`。首次训练没有旧 World Model 时，将 `warm_start_model` 设为 `null`。run 名称和目录统一为 `q1-robot-supermimic-<YYYYMMDD-HHMMSS>`，`logs/latest` 指向当前运行；检查点为 `logs/latest/{best,last}.pt`。World Model 按实际想象步数验收，Critic 先学习真实轨迹回报；每个 Actor 小步更新都经动作偏移检查和完整回合评估，退步时只回滚 Actor。默认不因连续回滚退出训练。训练使用 CUDA，在线采集和评估使用 CPU 多进程。GPU 显存预算和预留量可在配置中设置；空闲显存不足会退出。

续训时在配置中改用新的 `output`，然后执行：

```bash
python train.py --config configs/soccer.json --resume logs/latest/last.pt
```

续训需要保留原运行目录里的 `online/` 数据。W&B：先 `wandb login`，再在配置中设置 `"wandb": true`。

GUI 回放训练后的学生（支持 `.pt` / `.ckpt`）：

```bash
python play_mujoco.py --ckpt logs/latest/best.pt

# 同时逐回合保存 MP4。
python play_mujoco.py --ckpt logs/latest/best.pt --out-video video
```

GUI 默认跟随机器人，按 `F` 切换相机视野显示。回放使用 CPU 推理、训练时的目标角与力矩限幅；进球、摔倒、出界或超时后自动开始下一回合。
入口会在当前环境缺少依赖时自动查找本机 `q1-supermimic` / `q1-kick` 环境；`--ckpt` 必须指向实际存在的文件。

## Docker

```bash
docker build -t q1-supermimic .
docker run --rm --gpus all --ipc=host \
  -v "$PWD/expert/checkpoints:/workspace/expert/checkpoints:ro" \
  -v "$PWD/expert/datasets:/workspace/expert/datasets" \
  -v "$PWD/logs:/workspace/logs" \
  q1-supermimic python train.py --config configs/soccer.json
```

## Pole weaving

The independent Q1 S-route training, headless evaluation, and playback flow is
documented in [docs/pole-weaving.md](docs/pole-weaving.md).

## 代码

- `3rdparty/SuperMimic/`：厂家源码，固定版本 `118338b`，保持原样。
- `q1_soccer/`：Q1 V3、球动力学、专家数据与训练入口的适配层。继承厂家的 SuperMimic 更新核心。
- `expert/`：专家推理、回合采集和验证，详见 [说明](expert/README.md)。

策略输入保持专家的 79 维观测 × 50 帧历史，包括视野、噪声和延迟；真实状态仅用于 World Model、Critic 和参考监督。这里的视觉接口是模拟球检测特征，尚未接 RGB 检测器。World Model、Critic 和动作模仿都只使用严格成功的训练回合；学生在线回合未成功则直接丢弃。验证回合仅用于评估，不参与拟合。

厂家原项目面向 Q1 V2；本项目使用专家配套 V3 模型。厂家源码要求的完整环境见其 README，本项目适配层在 Python 3.11、PyTorch 2.9、MuJoCo 3.17 环境验证。

厂家源码采用 MIT；本项目原有代码与专家文件的许可证见 [LICENSE](LICENSE) 及对应文件。
