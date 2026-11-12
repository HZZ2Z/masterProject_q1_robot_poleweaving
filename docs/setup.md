# 环境搭建

本指南用于首次配置 Q1 机器人足球课程的统一 Python 环境。两门课程共用 Python 3.11、MuJoCo 和 PyTorch 基础依赖；各课程所需的模型、数据和训练资源请查看对应课程 README。

## 系统要求

- Linux
- Conda 或 Mamba
- Git，以及用于下载仓库子模块的网络连接

默认通过 `cuda:0` 运行的 S 型绕杆训练需要可用的 NVIDIA GPU 和兼容 CUDA 12.8 的驱动；足球踢球正式训练同样使用 CUDA。环境依赖安装的是 CUDA 12.8 版 PyTorch。没有可用 CUDA 设备时，仍可运行足球踢球的 CPU 冒烟训练。足球踢球与 S 型绕杆的训练还需要各自课程页列出的数据或 checkpoint 资产。

## 安装步骤

在仓库根目录打开终端，初始化项目使用的子模块：

```bash
git submodule update --init --recursive
```

创建 Conda 环境。以下命令使用清华 PyPI 镜像安装普通 Python 包；PyTorch CUDA wheel 仍从 `requirements.txt` 声明的 PyTorch 官方索引下载：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple conda env create -f environment.yml
```

如果不能访问清华 PyPI 镜像，可以移除 `PIP_INDEX_URL=...`，使用默认 PyPI 源：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate env_q1
```

在仓库根目录将课程与共享 Python 包以可编辑模式安装。源码仍从 `courses/` 和 `shared/` 加载，修改后不需要重新安装：

```bash
python -m pip install --no-deps -e .
```

## 检查安装

确认 Python、项目包、PyTorch、MuJoCo 和 Warp 均可导入，并查看 PyTorch 是否识别 CUDA 设备：

```bash
python -c "import sys, torch, mujoco, mujoco_warp, warp, q1_soccer, pole_task, envs, utils, expert; print('Python:', sys.version.split()[0]); print('Project packages: OK'); print('PyTorch:', torch.__version__); print('MuJoCo packages: OK'); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` 表示当前 PyTorch 没有发现可用的 CUDA 设备，不一定代表安装失败；你仍可运行足球踢球的 CPU 冒烟训练。需要 `cuda:0` 的训练与预览命令要求 CUDA 显示为可用。

完成检查后，按已选课程 README 的首次体验步骤继续。如果还没有选课，请先返回[项目主页](../README.md)选择一门课程，再打开对应课程页。

## 常见问题

### 环境创建时下载依赖失败

先确认网络可以访问所配置的 Conda 和 PyPI 源。若清华 PyPI 镜像不可用，可改用默认 PyPI 源；PyTorch wheel 使用独立的官方索引，相关下载也需要可访问。

### CUDA 不可用

检查机器是否装有 NVIDIA GPU、驱动是否正常，以及驱动是否支持当前 CUDA 12.8 PyTorch wheel。没有 CUDA 设备时可以先完成包导入检查和足球踢球的 CPU 冒烟训练；不要使用需要 `cuda:0` 的训练与预览命令。

### 子模块没有内容

在仓库根目录重新执行 `git submodule update --init --recursive`，确认网络可访问项目配置的上游仓库，然后再创建或更新 Conda 环境。

### 不知道课程需要哪些模型或数据

先看对应课程 README 的“资源准备”或“首次体验前”小节。S 型绕杆的资产与复现细节见[杆间绕行复现说明](pole-weaving.md)。课程专用资源不一定随仓库提交；如果课程说明需要维护者提供，请按该页给出的方式联系维护者。
