# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Q1 Robot Soccer" \
      org.opencontainers.image.source="https://github.com/MohismLab/Q1-Robot-Soccer" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    WANDB_MODE=disabled \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/q1-cache

# EGL for NVIDIA offscreen rendering, GLFW for optional desktop playback,
# and OSMesa for software rendering when no GPU/display is available.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       libegl1 \
       libgl1 \
       libglfw3 \
       libosmesa6 \
       libx11-6 \
       libxrandr2 \
       libxinerama1 \
       libxcursor1 \
       libxi6 \
       libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt ./requirements.txt
COPY courses/soccer-4v4/third_party/mappo ./courses/soccer-4v4/third_party/mappo
# The base image supplies PyTorch 2.9.0 with CUDA 12.8. imageio[ffmpeg]
# supplies the H.264 encoder used by --out-video.
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .
RUN python -m pip install --no-deps -e . \
    && mkdir -p logs video

# Override the command for smoke runs, CPU checks, evaluation or playback.
CMD ["python", "courses/soccer-4v4/train.py", "--backend", "warp", "--device", "cuda:0", "--num-envs", "1024", "--no-wandb"]
