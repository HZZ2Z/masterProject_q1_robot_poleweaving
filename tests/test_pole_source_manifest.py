import hashlib
import json

import pytest

from pole_task.source import verify_source_manifest


def test_source_manifest_verifies_registered_artifact_hashes(tmp_path):
    project = tmp_path / "project"
    source = project / "source"
    manifest_dir = project / "courses/pole-weaving/pole_task"
    manifest_dir.mkdir(parents=True)
    source.mkdir()
    artifact = source / "checkpoint.pt"
    artifact.write_bytes(b"warmup checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = manifest_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_root": "../../../source",
        "source_revision": "fixture-revision",
        "warmup_checkpoint_id": "warmup_checkpoint",
        "final_checkpoint_reference_id": "passed_actor",
        "tensor_shapes": {},
        "artifacts": [{
            "id": "warmup_checkpoint",
            "path": "checkpoint.pt",
            "sha256": digest,
        }],
    }))

    report = verify_source_manifest(manifest_path)

    assert report["verified_artifacts"] == ["warmup_checkpoint"]
    assert report["source_root"] == str(source.resolve())


def test_source_manifest_rejects_hash_mismatch_before_returning(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "checkpoint.pt").write_bytes(b"unexpected checkpoint")
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_root": ".",
        "source_revision": "fixture-revision",
        "warmup_checkpoint_id": "warmup_checkpoint",
        "final_checkpoint_reference_id": "passed_actor",
        "tensor_shapes": {},
        "artifacts": [{
            "id": "warmup_checkpoint",
            "path": "checkpoint.pt",
            "sha256": "0" * 64,
        }],
    }))

    with pytest.raises(ValueError, match="warmup_checkpoint.*SHA-256 mismatch"):
        verify_source_manifest(manifest_path)


def test_source_manifest_rejects_artifact_paths_outside_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "checkpoint.pt"
    outside.write_bytes(b"outside source")
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_root": "source",
        "source_revision": "fixture-revision",
        "warmup_checkpoint_id": "warmup_checkpoint",
        "final_checkpoint_reference_id": "passed_actor",
        "tensor_shapes": {},
        "artifacts": [{
            "id": "warmup_checkpoint",
            "path": "../checkpoint.pt",
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }],
    }))

    with pytest.raises(ValueError, match="escapes the registered source root"):
        verify_source_manifest(manifest_path)
