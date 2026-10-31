"""Verify the pinned manufacturer source, also in Docker without .git metadata."""

import hashlib
import json
from pathlib import Path
from . import VENDOR


def identity():
    document = json.loads(Path(__file__).with_name("upstream.json").read_text())
    for name, expected in document["files_sha256"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Manufacturer source differs from pinned commit: {name}")
    return document
