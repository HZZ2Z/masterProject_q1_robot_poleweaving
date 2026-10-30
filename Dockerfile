# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Q1 Robot Soccer Supermimic" \
      org.opencontainers.image.source="https://github.com/MohismLab/Q1-Robot-Soccer/tree/SuperMimic" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/q1-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates libegl1 libgl1 libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check
COPY . .
RUN mkdir -p logs

CMD ["python", "train.py", "--help"]
