# Q1 Robot Soccer

启元 PrimeBot Q1 V3 足球训练项目。仿真使用 MuJoCo Warp，训练使用 PyTorch、PPO 和 AMP。

## 使用

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
# 环境检查：运行两轮
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 8 --max_iterations 2 --save_path logs/q1_smoke --no_wandb

# 训练
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 4096 --save_path logs/q1 --no_wandb
```

`--num_envs` 设置并行环境数，按可用显存调整。

权重目录：`logs/q1/logs/<时间>/nn/`。`last.pth` 为最新权重，`best.pth` 按训练指标选取。

启用 W&B：执行 `wandb login`，去掉 `--no_wandb`。

### 回放与录像

将 `CHECKPOINT` 替换为本地权重路径。

```bash
CHECKPOINT="logs/q1/logs/你的训练时间/nn/best.pth"

# 窗口回放并录像
python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --out-video "$PWD/video"

# 无窗口录像
MUJOCO_GL=egl python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --headless --steps 5000 --out-video "$PWD/video"
```

每个回合保存一个 MP4，分辨率 1280×720，帧率 50 FPS。关闭窗口或按 Ctrl+C 时保存当前回合。

去掉 `--out-video` 可关闭录像。窗口中按 **F** 显示或隐藏相机视锥。

### Docker

宿主机需要 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。

```bash
docker build -t q1-robot-soccer:local .
mkdir -p logs video

# 训练
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs" \
  q1-robot-soccer:local \
  python train.py --task Q1 --headless --device cuda:0 \
    --num_envs 4096 --save_path logs/q1 --no_wandb

# 无窗口录像
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs:ro" \
  -v "$PWD/video:/workspace/video" \
  q1-robot-soccer:local \
  python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
    --headless --steps 5000 --out-video /workspace/video
```

容器中启用 W&B：在宿主机设置 `WANDB_API_KEY`，为 `docker run` 添加 `-e WANDB_MODE=online -e WANDB_API_KEY`，去掉 `--no_wandb`。

## 来源

- 方法：[Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots](https://humanoid-kick.github.io/)
- Q1 模型：[PrimebotTech/primebot-q1-urdf](https://github.com/PrimebotTech/primebot-q1-urdf)
- 许可证：[BSD-3-Clause](LICENSE)
