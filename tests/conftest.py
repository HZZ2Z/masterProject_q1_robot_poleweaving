"""Expose canonical source directories to the test process."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    PROJECT_ROOT / "courses/pole-weaving",
    PROJECT_ROOT / "shared/supermimic",
    PROJECT_ROOT / "shared/q1-mujoco",
)
for source_path in reversed(SOURCE_PATHS):
    sys.path.insert(0, str(source_path))
