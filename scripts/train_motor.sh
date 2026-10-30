#!/usr/bin/env bash
# Local RTX 5090, shared with a 14 GiB 4v4 job. Motor foundation only.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
Q1_PYTHON="${Q1_PYTHON:-python}"
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  "$Q1_PYTHON" -u train.py \
  --iterations 20000 --num-envs 2048 \
  --model-batch-size 2048 --policy-batch-size 512 \
  --model-width 1024 --policy-width 512 \
  --warmup-batches 200 --model-batches 10 \
  --evaluation-interval 5 --evaluation-episodes 16 --save-interval 10 \
  --exploration-std .01 --initial-perturbation .005 --actor-lr .00001 \
  --threads 8 --memory-fraction .35 --wandb "$@"
