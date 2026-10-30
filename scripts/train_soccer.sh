#!/usr/bin/env bash
# Experimental downstream integration, shared RTX 5090. Motor snapshot required.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
Q1_PYTHON="${Q1_PYTHON:-python}"
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "$Q1_PYTHON" -u train_soccer.py "$@"
