#!/usr/bin/env bash
set -euo pipefail
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$script_path")/../../.."
if [[ -n "${Q1_4V4_PYTHON:-}" ]]; then
  q1_python="$Q1_4V4_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX%/*}/q1-kick-4v4/bin/python" ]]; then
  # Prefer the project's sibling environment, even when q1-kick is active.
  q1_python="${CONDA_PREFIX%/*}/q1-kick-4v4/bin/python"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE%/bin/conda}/envs/q1-kick-4v4/bin/python" ]]; then
  q1_python="${CONDA_EXE%/bin/conda}/envs/q1-kick-4v4/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/envs/q1-kick-4v4/bin/python" ]]; then
  q1_python="$CONDA_PREFIX/envs/q1-kick-4v4/bin/python"
elif [[ -x /home/kemove/miniconda3/envs/q1-kick-4v4/bin/python ]]; then
  q1_python=/home/kemove/miniconda3/envs/q1-kick-4v4/bin/python
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  q1_python="$CONDA_PREFIX/bin/python"
else
  q1_python="$(command -v python3)"
fi
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH "$q1_python" "$@"
