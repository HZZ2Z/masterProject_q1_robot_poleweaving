"""Provenance and integrity checks for the migrated source task artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_robot_xml(manifest_path: Path) -> Path:
    """Resolve and hash-check the registered source robot XML.

    Scene composition needs the source robot without hashing the large policy
    checkpoints, so this verifies only the registered robot contract entry.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    source_root_value = Path(manifest["source_root"])
    source_root = source_root_value if source_root_value.is_absolute() else manifest_path.parent / source_root_value
    source_root = source_root.resolve()
    contract = manifest["robot_contract"]
    path = (source_root / Path(contract["xml"])).resolve()
    if source_root not in path.parents:
        raise ValueError("Registered source robot XML escapes the source root")
    if not path.is_file():
        raise ValueError(f"Registered source robot XML is missing: {path}")
    if _sha256(path) != str(contract["xml_sha256"]).lower():
        raise ValueError(f"Source robot XML SHA-256 mismatch: {path}")
    return path


def verify_source_manifest(manifest_path: Path) -> dict:
    """Verify every registered source artifact without loading policy weights."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    source_root_value = Path(manifest["source_root"])
    source_root = source_root_value if source_root_value.is_absolute() else manifest_path.parent / source_root_value
    source_root = source_root.resolve()
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Source manifest must register at least one artifact")

    verified = []
    artifact_paths = {}
    seen_ids = set()
    for artifact in artifacts:
        artifact_id = str(artifact["id"])
        if artifact_id in seen_ids:
            raise ValueError(f"Duplicate source artifact id: {artifact_id}")
        seen_ids.add(artifact_id)
        relative_path = Path(artifact["path"])
        if relative_path.is_absolute():
            raise ValueError(f"Source artifact {artifact_id} path must be relative")
        path = (source_root / relative_path).resolve()
        if path != source_root and source_root not in path.parents:
            raise ValueError(f"Source artifact {artifact_id} escapes the registered source root")
        if not path.is_file():
            raise ValueError(f"Source artifact {artifact_id} is missing: {path}")
        expected_hash = str(artifact["sha256"]).lower()
        if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError(f"Source artifact {artifact_id} has an invalid SHA-256 value")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source artifact {artifact_id} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        verified.append(artifact_id)
        artifact_paths[artifact_id] = str(path)

    return {
        "manifest": str(manifest_path),
        "source_root": str(source_root),
        "source_revision": manifest["source_revision"],
        "verified_artifacts": verified,
        "verified_artifact_count": len(verified),
        "artifact_paths": artifact_paths,
        "warmup_checkpoint_id": manifest["warmup_checkpoint_id"],
        "final_checkpoint_reference_id": manifest["final_checkpoint_reference_id"],
        "tensor_shapes": manifest["tensor_shapes"],
    }
