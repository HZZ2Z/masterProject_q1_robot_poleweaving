"""Project-owned pole weaving task for the Q1 soccer scene."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "pole_task" / "data"
VENDOR = ROOT / "3rdparty/SuperMimic"
if not (VENDOR / "src/supermimic/algorithm.py").is_file():
    raise ImportError("Run git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR / "src"))
