#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
Q1_PYTHON="${Q1_PYTHON:-/home/kemove/miniconda3/envs/q1-kick/bin/python}"
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH "$Q1_PYTHON" -m pytest -q "$@"
