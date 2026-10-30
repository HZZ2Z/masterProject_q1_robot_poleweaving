# Q1 Robot Soccer

启元 PrimeBot Q1 V3 踢球训练与回放，基于 [humanoid-kick](https://gitee.com/chaomingsanhua/humanoid-kick) 适配，使用 MuJoCo Warp、PyTorch、PPO + AMP，无需 Isaac Gym。

## Quick Start

### 安装

需要 Linux、Conda 和支持 CUDA 12.8 的 NVIDIA 驱动。

```bash
git clone git@github.com:MohismLab/Q1-Robot-Soccer.git
cd Q1-Robot-Soccer
conda env create -f environment.yml
conda activate q1-kick
```

### 训练

```bash
# 先跑两轮检查环境。
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 8 --max_iterations 2 --save_path logs/q1_smoke --no_wandb

# 正式训练，按显存容量调整环境数。
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 4096 --save_path logs/q1 --no_wandb
```

权重保存在 `logs/q1/logs/<时间>/nn/`，最新权重为 `last.pth`，最佳权重为 `best.pth`。需要 W&B 时先执行 `wandb login`，再去掉 `--no_wandb`。

### 回放与录像

仓库不包含训练权重，请替换下面的 checkpoint 路径。运行前确认已进入 `q1-kick` 环境。

```bash
CHECKPOINT="logs/q1/logs/你的训练时间/nn/best.pth"

# 打开窗口，每个回合保存一个 MP4。
python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --out-video "$PWD/video"

# 无窗口录像。
MUJOCO_GL=egl python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --headless --steps 5000 --out-video "$PWD/video"
```

录像为 1280×720、50 FPS；关闭窗口或 Ctrl+C 也会保存当前回合。不需要录像时去掉 `--out-video`，按 **F** 切换相机视锥。

### Docker

宿主机需要 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。

```bash
docker build -t q1-robot-soccer:local .
mkdir -p logs video

# 训练。
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs" \
  q1-robot-soccer:local \
  python train.py --task Q1 --headless --device cuda:0 \
    --num_envs 4096 --save_path logs/q1 --no_wandb

# 无窗口录像，使用上面设置的 CHECKPOINT。
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs:ro" \
  -v "$PWD/video:/workspace/video" \
  q1-robot-soccer:local \
  python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
    --headless --steps 5000 --out-video /workspace/video
```

镜像默认禁用 W&B；需要启用时增加 `-e WANDB_MODE=online -e WANDB_API_KEY`，并去掉 `--no_wandb`。日志、权重、录像和凭据不打包进镜像。

## 来源

- 方法：[Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots](https://humanoid-kick.github.io/)
- Q1 模型：[PrimebotTech/primebot-q1-urdf](https://github.com/PrimebotTech/primebot-q1-urdf)
- 许可证：[BSD-3-Clause](LICENSE)
