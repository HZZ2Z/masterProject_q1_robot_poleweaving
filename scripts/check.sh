#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
Q1_PYTHON="${Q1_PYTHON:-python}"
exec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH "$Q1_PYTHON" -m pytest -q "$@"
