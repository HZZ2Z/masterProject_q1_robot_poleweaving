# Q1 Humanoid Kick

启元 PrimeBot Q1 V3 的视觉驱动踢球训练项目。基于 [humanoid-kick](https://gitee.com/chaomingsanhua/humanoid-kick)（起始提交 `9836813`）迁移，保留 PPO + AMP、历史观测编码/重建、虚拟视觉和踢球奖励。Q1 使用 **MuJoCo Warp 在 GPU 上并行仿真，PyTorch 在同一 GPU 上学习**，无需安装 Isaac Gym。

参考论文：[Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots](https://humanoid-kick.github.io/)。这份 Q1 适配尚未完成策略收敛验证，也不提供可直接控制真机的策略。

## Quick Start

本项目对应 Q1 的 PPO + AMP 训练、MuJoCo 回放和 MP4 录制。机器人资源和参考动作已包含在仓库，无需 Isaac Gym。**训练日志、checkpoint 和录像不随代码发布**；回放需提供自己的 Q1 checkpoint，不能直接使用本文作者机器上的日志路径。

### 1. 克隆与安装（Conda）

GPU 训练需要 Linux、NVIDIA GPU 和兼容 CUDA 12.8 的驱动。已验证的本机环境为 RTX 5090 / 32GB、驱动 595.99.02。无 GPU 可用 CPU 做小规模检查或回放。

```bash
git clone git@github.com:MohismLab/Q1-Robot-Soccer.git
cd Q1-Robot-Soccer

conda env create -f environment.yml
conda activate q1-kick

python -c "import torch, mujoco; print('Torch:', torch.__version__, 'MuJoCo:', mujoco.__version__, 'CUDA:', torch.cuda.is_available())"
```

后续所有 `python` 命令都在仓库根目录、`q1-kick` 环境中执行。若出现 `No module named mujoco`，先检查终端是否仍在 `(base)` 环境。Linux 精简系统若缺少渲染库，可安装：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libglfw3 libosmesa6
```

### 2. 小规模检查与训练

先跑 2 轮验证依赖、资产和 GPU 后端；这只检查流程，不代表策略已学会踢球：

```bash
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 8 --max_iterations 2 --save_path logs/q1_smoke --no_wandb
```

CPU 检查将 `--device cuda:0` 改为 `--device cpu`，并加 `--num_threads 2`。正式训练可逐步扩大环境数；当前本机配置为 4096 环境，显存预算为单个训练进程占总显存的 50%：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 4096 --save_path logs/q1 --no_wandb
```

日志输出到 `logs/q1/logs/<时间>/`，权重位于该目录的 `nn/` 下，包含 `last.pth` 和 `best.pth`。首次 GPU 运行会编译 Warp 内核。需要 W&B 时先执行 `wandb login`，再去掉 `--no_wandb`；本地 TensorBoard 查看方式见下文。

### 3. 回放和每回合录像

用实际的 checkpoint 路径替换 `CHECKPOINT`；新克隆的仓库中没有预训练权重。

```bash
CHECKPOINT="logs/q1/logs/<你的训练时间>/nn/best.pth"

# 有窗口：边回放边按回合保存 MP4。
python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --out-video "$PWD/video"

# 无窗口：运行 5000 个物理步，结束时也保存当前未完成回合。
MUJOCO_GL=egl python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
  --headless --steps 5000 --out-video "$PWD/video"

# 固定场景评估。
python evaluate_q1.py --checkpoint "$CHECKPOINT" \
  --output logs/q1/evaluations/result.json
```

录像默认 1280×720、50 FPS；每回合一个文件，名称包含启动时间、回合编号及 `goal/out/fall/timeout`。关闭窗口或达到步数上限保存 `stopped.mp4`，Ctrl+C 保存 `interrupted.mp4`。正在编码的文件以 `.partial.mp4` 结尾，结束后自动重命名。省略 `--out-video` 时不录像。

### 4. Docker（GPU 训练与无头录像）

宿主机需安装 Docker、NVIDIA 驱动和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，使 `docker run --gpus all` 可用。镜像基于 PyTorch 2.9.0 / CUDA 12.8，默认使用 EGL 无头渲染和禁用 W&B；宿主机无需另装 Conda。

```bash
docker build -t q1-robot-soccer:local .
mkdir -p logs video

# GPU 冒烟检查；以宿主用户写日志，避免生成 root 所有的文件。
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs" \
  q1-robot-soccer:local \
  python train.py --task Q1 --headless --device cuda:0 \
    --num_envs 8 --max_iterations 2 --save_path logs/q1_smoke --no_wandb

# 正式训练：按显存容量调整 --num_envs。
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs" \
  q1-robot-soccer:local \
  python train.py --task Q1 --headless --device cuda:0 \
    --num_envs 4096 --save_path logs/q1 --no_wandb

# 使用已挂载 logs 中的权重，录像写回宿主机 video/。
CHECKPOINT="logs/q1/logs/<你的训练时间>/nn/best.pth"
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/logs:/workspace/logs:ro" \
  -v "$PWD/video:/workspace/video" \
  q1-robot-soccer:local \
  python play_mujoco.py --task Q1 --checkpoint "$CHECKPOINT" \
    --headless --steps 5000 --out-video /workspace/video
```

`logs/`、`video/`、权重和本机凭据通过 `.dockerignore` 排除，运行时挂载所需目录。若权重不在 `logs/` 中，请另行挂载其所在目录。无 NVIDIA GPU 时可省略 `--gpus all`、训练指定 `--device cpu`；软件渲染录像同时覆盖 `-e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa`。Docker 默认不打开桌面窗口，交互式回放推荐使用上面的本机 Conda 命令。

容器启用 W&B 时，给 `docker run` 增加 `-e WANDB_MODE=online -e WANDB_API_KEY`，在宿主环境设置 `WANDB_API_KEY`，并去掉训练命令的 `--no_wandb`；使用自定义 W&B 服务时还需传入 `-e WANDB_BASE_URL`。凭据仅在运行时传入。

## 环境与实现

Python 3.11、PyTorch 2.9.0 / CUDA 12.8、MuJoCo 3.17.0、mujoco-warp 3.17.0、warp-lang 1.18.0。GPU 训练通过 Torch/Warp 共享 CUDA stream 和零拷贝状态，并用 CUDA Graph 执行物理。`--device cpu` 使用原生 MuJoCo Batch 作测试与预览参考；`--num_threads` 只影响 CPU 参考后端。首次运行会编译 Warp 内核。W&B 与本地 TensorBoard 每轮实时记录。

W&B 项目为 `mohismlab-robotics/q1-humanoid-kick`。每完成一轮训练就实时提交一条包含该轮回报、成功率、摔倒率、loss 和速度的记录，横轴为 `iteration`。网络可能带来数秒延迟，不需要等训练结束。凭据通过 `wandb login` 保存在用户目录，禁止写入配置或仓库。需要代理时给启动命令设置 `HTTPS_PROXY=http://127.0.0.1:10808`。可在 `runner.use_wandb` 关闭云端记录。

## 训练与回放

先跑少量迭代检查环境：

```bash
python train.py --task Q1 --headless --num_envs 8 --num_threads 2 \
  --max_iterations 2 --save_path logs/q1_smoke
```

当前正式配置为 4096（训练进程显存预算 50%） 个并行环境，每轮采集 24 步；PPO 使用 4 个小批次：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python train.py --task Q1 --headless --device cuda:0 \
  --num_envs 4096 --save_path logs/q1
```

默认上限沿用上游的 20,000 次迭代，每轮采集 `4096 × 24 = 98304` 个样本，PPO 分为 4 个小批次。RTX 5090 上完整 Warp 更新约 7.3–7.8 万步/秒；9728 环境约 8.0–8.3 万步/秒，整卡显存约 30–31GB（含其他进程），启用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 减少碎片。10240 曾完成短测，但在后续启动时因碰撞临时缓存显存不足退出，该数据来自历史容量测试；当前按半卡预算使用 4096。16384 环境叠加物理、历史观测和 PPO 缓存会超出 32GB。监控 `performance/cuda_peak_allocated_gib`（Torch 分配）和 `performance/cuda_reserved_gib`；它们不包含 Warp 内存，全部显存请同时看 `nvidia-smi`。

稳定性修复加入站立课程：保留编码器和 actor 主干，一次性重置动作输出头及 critic，从小噪声的中立站姿学习。达到最早切换轮次、至少 80% 的环境已连续存活 5 秒，且近期重置试次的 5 秒存活率达到门槛，才进入下一阶段。切换时保留正在运行的轨迹，并保存 `standing.pth`；未来重置逐渐混入运动参考帧，不会一次全部替换为动态姿态。

课程版本 2 按稳定性调整踢球奖励和 AMP：每 20 轮观察一个窗口，再决定增加、保持或降低难度；单次最多增加 0.02，扰动力也随难度缩放。`policy_bootstrap_version`、课程版本和进度随 checkpoint 保存，恢复时不会重复重置动作头。

Q1 奖励版本 4：用官方脚部碰撞网格的包围盒与球半径计算三维表面间隙（保守的接近判定，不等于精确接触检测）。间隙小于 0.12m 进入近球区，超过 0.16m 才退出，防止阈值抖动反复重置准备时间；允许 1.5 秒准备，之后 1.5 秒内逐渐施加最多 -2/秒的近球停滞惩罚，并将该环境的 AMP 奖励逐渐降到零。只有球确实向球门运动（>0.05m/s）才能重置近球计时，脚部空摆或绕球不能重置。远球阶段按完整 2 秒窗口比较机器人到窗口起始球位置的净距离变化；净接近不足 4cm 时累计停滞，不能用单步晃动清零。持续停滞后惩罚逐渐达到 -1/秒，AMP 系数也同步降到零；没有原来骨盆距球 1m 内的豁免。

踢球权重为正踢 `2`、侧踢 `8`。脚球表面间隙 <0.03m、姿态直立且未跌倒时，以球朝球门的实际位移速度（上限 2m/s）为共同基础，再按最近脚在自身坐标系内的前向/横向速度占比分配两个奖励。斜向踢球分摊权重，不重复获得两项完整奖励；脚几乎不动、静止球和空摆腿不奖励。该邻近条件不是精确碰撞标签。保留球门距离进展与进球奖励。新奖励版本记录进 checkpoint；恢复旧策略时保留网络和优化器，前后 reward 不宜直接比较。

当前初始化固定分为三个环境组：25% 在门线以内的矩形场地均匀采样（球心 x∈[-6.9,6.9]、y∈[-4.4,4.4]，为 0.1m 球半径留出余量）；25% 在场地中心半径 3.5m 的圆内按面积均匀采样；50% 将机器人放为朝向球、脚球间隙约 10–25cm 的中立安全姿态，这组的球仍在中心圆内。球与原机器人重叠时拒绝重采样。按环境编号固定分组，4096 环境对应 1024/1024/2048，部分重置不改变组别。`near_start.adaptive=false`，比例和近球距离不会自动扩张；保留分组成功率和固定评估用于比较。回放按同样的 25/25/50 概率选择初始化组，固定评估显式关闭训练出生分布。

`runner.gpu_memory_fraction_limit=0.5` 用 NVML 查询当前训练 PID 的 GPU 占用（含 Warp、Torch 和 CUDA 缓存），不将其他任务计入本训练预算。超限时在本轮结束保存并停训；当前使用 4096×24 留出余量。监控 `performance/gpu_process_used_gib`；`performance/gpu_total_used_gib` 仅显示整卡占用，不作为停训依据。

固定评估使用原生 MuJoCo 标称模型、确定性策略、固定种子和 24 个场景（骨盆距球 0.4/0.8/1.5/3m、朝向误差 -60/0/60°、两种球位置），每例最多 20 秒，不随训练课程改变。触球读取真实脚球碰撞，推进统计触球后的球门距离改善；记录到位率、触球率、有效踢球率、进球率、触球后摔倒率和近球停滞。近距离可能落入当前虚拟相机盲区，这些难例保留在评估中。每 500 轮保存存档时自动在 CPU 后台评估，结果位于本次 run 的 `evaluations/`，TensorBoard/W&B 的 `evaluation/*` 按被评估的 checkpoint 轮次记录，包含按距离分组的结果。它是固定仿真基准，不是训练指标或真机成绩。

第 6564 轮基线（上述 24 场景、每例 20 秒）：触球 17/24，进球 7/24；0.4m 的 6 个近距离场景均未触球。这些数字用于后续同条件比较，不能代替训练成功率。

```bash
python evaluate_q1.py --checkpoint <Q1-checkpoint路径> \
  --output logs/q1/evaluations/result.json
```

关注 `episode/duration_s`、`stability/active_over_5s`、`stability/reset_cohort_survival_5s`、`stability/base_height_mean`、两项 `episode/termination_*_rate` 和 `curriculum/*`。奖励版本 4 记录 `episode/near_ball_s`、`episode/near_ball_stall_s`、`episode/far_ball_stall_s`、`episode/ball_forward_distance`，以及 `soccer/*` 的脚球间隙、近球/停滞比例与 AMP 系数。重置试次的存活率把 5 秒前的失败也计入，避免只看长寿环境产生偏差。`fall_rate` 统计已结束回合中的摔倒比例；尚无回合结束时不写入虚假的零值。终止高度和速度阈值保持不变。

训练成功率要求球入门后机器人仍未跌倒。随机给球加速度和随机换球已关闭，避免环境把球送进球门也被计为策略成功。课程版本 1 的短暂成功率峰值受此干扰，不能当作已学会踢球的证据；当前曲线仍是训练统计，不是固定测试集成绩。

没有可用 GPU 时使用 `--device cpu`。训练日志位于 `logs/q1/logs/<时间>/`，checkpoint 在其 `nn/` 下，每 100 次迭代及训练结束时保存 `model_<轮次>.pth`。

`last.pth` 指向最近保存的快照；`best.pth` 在已保存的快照中选择最近 100 个有回合结束的迭代的平均成功率最高者，平分时比较平均回报。两者是符号链接，不重复占用权重空间。选择结果记录于 `nn/checkpoint_selection.json`。这些来自训练指标，不代表独立测试集成绩。

`--checkpoint` 加载同结构的 Q1 存档，并恢复训练轮次、两个优化器和 AMP 归一化统计。早期仅含网络权重的存档会明确提示优化器/归一化统计重新初始化，并从文件名恢复轮次。回放缓冲区和仿真场景重新初始化；这不是仿真轨迹的逐位续跑。SIGINT/SIGTERM 会在当前迭代完成后保存退出。

```bash
tensorboard --logdir logs/q1
python play_mujoco.py --task Q1 --checkpoint <Q1-checkpoint路径>
python play_mujoco.py --task Q1 --checkpoint <Q1-checkpoint路径> --headless --steps 5000
python play_mujoco.py --task Q1 --initial-state  # 冻结的训练初始状态，无需权重
```

在训练机器的浏览器打开 http://127.0.0.1:6006/，查看 Scalars 下的 `reward`、`success_rate`、`fall_rate`、各项 loss 和 `performance/env_steps_per_second`。TensorBoard 是独立服务，关闭网页不会停止训练。若训练进程早于 best/last 功能升级启动，可用 `python -m utils.recorder --watch-run logs/q1/logs/<时间>` 为该进程的存档建立索引，无需重启训练；后续新启动的训练直接在保存时更新索引。

使用 `--out-video <目录>` 时，每个回合分别流式保存为 MP4（1280×720、50 FPS），目录自动创建。例如：

```bash
python play_mujoco.py --task Q1 \
  --checkpoint "$CHECKPOINT" \
  --out-video "$PWD/video"
```

文件名包含启动时间、回合编号和结束原因（`goal`、`out`、`fall`、`timeout`）。终止画面在重置前写入；关闭窗口或达到 `--steps` 上限时，当前未完成回合保存为 `stopped.mp4`，Ctrl+C 时为 `interrupted.mp4`。编码中的临时文件以 `.partial.mp4` 结尾，完成后重命名，不覆盖之前启动的录像。录像跟随回放视角及 F 键视锥开关；也支持 `--headless`，无显示环境可设置 `MUJOCO_GL=egl`。

回放默认显示蓝色半透明相机视锥，使用与策略观测完全相同的相机姿态和 FOV；按 **F** 切换显示，`--no-frustum` 默认隐藏，`--frustum-range 2.5` 指定绘制长度（只是可视化长度，不是相机量程限制）。冻结初始状态也显示视锥。球场采用明亮天空、草地、完整边线与中圈；球门增加圆管门框和网格外观，足球采用可重新生成的黑白五/六边形立方体纹理。装饰几何不参与碰撞，原有球门碰撞范围保留。`scripts/prepare_q1.py` 统一生成场景与纹理。

省略回放的 `--checkpoint` 时，只查找 `logs/q1` 下的 Q1 模型。G1 的 13 自由度权重不能用于 Q1。交互查看统一使用 `play_mujoco.py`；批量训练不打开窗口。

## 项目结构

```text
train.py / play.py          复用上游训练、策略运行入口
play_mujoco.py              Q1 / G1 / T1 共用的单环境回放入口
envs/
  q1.py                    Torch 状态、PD、随机化、重置和 Q1 镜像
  Q1.yaml                  Q1 参数与训练配置
  t1.py                    上游奖励、虚拟视觉和任务逻辑；供 Q1 复用
utils/                     复用 PPO、AMP、网络、缓冲区与运动学工具
  physics.py               Warp GPU / 原生 MuJoCo CPU 适配与 CUDA Graph
resources/q1_description/  官方 URDF、mesh、生成的踢球场景和来源记录
data/q1/                   Q1 参考动作及迁移质量记录
scripts/
  prepare_q1.py            从官方仓库生成项目内模型资源
  retarget_q1_motions.py   从 G1 参考动作生成 Q1 动作
tests/test_q1.py            模型、运动学、镜像、批量仿真与重置检查
```

上游的 T1/G1 模型和训练任务保留作参考，训练仍依赖 Isaac Gym；Q1 入口不导入或依赖 Isaac Gym。Q1 使用统一的 `train.py` 和 `utils/runner.py`，没有另建训练算法。

## Q1 模型与动作

- 模型来自 [启元官方 URDF 仓库](https://github.com/PrimebotTech/primebot-q1-urdf)。完整 22 个自由度：双腿 12、腰部 2、双臂 8。URDF 保留官方连杆、惯量、碰撞、限位和 22/36 Nm 力矩上限；来源提交与文件哈希见 `resources/q1_description/SOURCE.json`。
- MuJoCo 场景从同一份 URDF 生成，加入足球、球门、平面球场和策略传感器。踝关节 PD 已校准为 K=60、D=2.5，名义中立姿态无策略控制连续站立 6 秒、无力矩饱和通过回归检查；相机外参、视场角仍是仿真假设，均未实机标定。
- 观测维度 79，critic 91，特权状态 14，AMP 59，动作 22。物理步长 2 ms，控制周期 20 ms。
- 参考动作通过足端姿态的有界逆运动学迁移，按 Q1 尺寸缩放位移，重新计算速度并处理足底穿地。策略控制全部 22 关节；AMP 仅约束原数据中有示教的双腿及腰部 yaw，不使用合成的手臂/腰部 roll 常量作为模仿目标。动作范围覆盖训练集中全部关节姿态，参考状态重置留有速度阈值余量。
- AMP 使用原始观测更新均值/方差，用标准差归一化。旧版错误统计和 discriminator 不迁移到新版；该语义变更通过 `amp_normalizer_version` 标记。动作尺度改变时会重标定输出层和动作噪声以保留物理目标，再清除受影响的策略优化器动量。
- 默认训练集有 22 段、4771 帧；两段转身动作因最大足端误差超过 3.5 cm 被排除。质量指标和源文件哈希见 `data/q1/retarget_manifest.json`。这是几何迁移的启动先验，未证明动力学可行性和踢球成功率。
- 当前 Warp 版本使用平面球场，支持质量/惯量、质心、摩擦、PD 增益、关节偏移、控制延迟、球检测噪声/丢失、外力与速度扰动。没有直接套用 PhysX 的 compliance/restitution 参数。终止时重置整套机器人和球状态。

资源和动作已放入项目，可独立训练。需要重新生成时：

```bash
python scripts/prepare_q1.py --source ../primebot-q1-urdf
python scripts/retarget_q1_motions.py
python -m pytest -q
```

默认测试只运行 Q1 检查；原有 `tests/test_g1_assets.py` 需要上游 Isaac Gym + Pinocchio 环境。

MuJoCo Warp 的碰撞求解与旧 CPU 后端存在数值差异，切换后单独记录新 run。加载旧权重与优化器后继续训练，回报或摔倒率可能有适应期；吞吐提升不代表策略已收敛。每轮检查接触/约束容量溢出，溢出会明确报错，避免静默丢失接触。

圆内采样以场地中心 `(0, 0)` 为圆心、半径 `3.5m`（两门相距 14m，圆直径为其一半 7m）；25/25/50 的具体出生分组见上文。

Q1 单环境回放使用同一 Motion 参考采样器和安全球出生范围；进球保持 50 个控制步，摔倒/出界/成功/60 秒超时均重置机器人和球。回放仍使用原生 MuJoCo 的标称模型，不等于 Warp 随机化训练的精确复现。
