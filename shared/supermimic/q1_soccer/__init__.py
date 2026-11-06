"""Soccer adapters for the unmodified PrimebotTech SuperMimic implementation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "3rdparty/SuperMimic"
if not (VENDOR / "src/supermimic/algorithm.py").is_file():
    raise ImportError("Run git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR / "src"))
