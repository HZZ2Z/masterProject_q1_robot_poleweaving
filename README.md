# Q1 Robot Soccer Supermimic

Q1 足球训练实验，使用世界模型和想象轨迹更新策略。

## Quick Start

### 安装

需要 Linux、Conda 和支持 CUDA 12.8 的 NVIDIA GPU 驱动。

```bash
git clone --branch SuperMimic git@github.com:MohismLab/Q1-Robot-Soccer.git
cd Q1-Robot-Soccer
conda env create -f environment.yml
conda activate q1-supermimic
```

### 训练

先训练动作基础，再启动踢球训练：

```bash
python train.py --iterations 200 --output logs/motor

python train_soccer.py --config configs/soccer_truth_joint.json \
  --motor-checkpoint logs/motor/last.pt --output logs/soccer
```

当前踢球配置使用球位置真值，联合更新目标编码器和动作解码器。配置位于 `configs/`，检查点保存为输出目录下的 `last.pt`。

启用 W&B：先执行 `wandb login`，再给训练命令加 `--wandb`。

### 继续训练

```bash
python train_soccer.py --config configs/soccer_truth_joint.json \
  --motor-checkpoint logs/motor/last.pt --output logs/soccer \
  --resume logs/soccer/last.pt
```

### Docker

宿主机需要 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。

```bash
docker build -t q1-supermimic:local .
mkdir -p logs

# 动作基础。
docker run --rm --gpus all --ipc=host \
  -v "$PWD/logs:/workspace/logs" q1-supermimic:local \
  python train.py --iterations 200 --output logs/motor

# 踢球训练。
docker run --rm --gpus all --ipc=host \
  -v "$PWD/logs:/workspace/logs" q1-supermimic:local \
  python train_soccer.py --config configs/soccer_truth_joint.json \
    --motor-checkpoint logs/motor/last.pt --output logs/soccer
```

日志、权重和凭据不打包进镜像。容器启用 W&B 时增加 `-e WANDB_API_KEY`，并给训练命令加 `--wandb`。

## 来源

- 方法：[SuperMimic](https://supermimic.github.io/)
- Q1 模型：[PrimebotTech/primebot-q1-urdf](https://github.com/PrimebotTech/primebot-q1-urdf)
- 许可证：[BSD-3-Clause](LICENSE)
