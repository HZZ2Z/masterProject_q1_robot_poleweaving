"""Project-owned pole weaving task for the Q1 soccer scene."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
COURSE_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT / "shared"
DATA = COURSE_ROOT / "pole_task" / "data"
VENDOR = ROOT / "3rdparty/SuperMimic"


def project_path(value):
    """Resolve a project-root-relative path independently of the working directory."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
if not (VENDOR / "src/supermimic/algorithm.py").is_file():
    raise ImportError("Run git submodule update --init --recursive")
sys.path.insert(0, str(VENDOR / "src"))
