#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${Q1_4V4_PYTHON:-}" ]]; then
  q1_python="$Q1_4V4_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != base ]]; then
  q1_python="$CONDA_PREFIX/bin/python"
elif [[ -x /home/kemove/miniconda3/envs/q1-kick-4v4/bin/python ]]; then
  q1_python=/home/kemove/miniconda3/envs/q1-kick-4v4/bin/python
else
  q1_python="$(command -v python3)"
fi
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH "$q1_python" "$@"
