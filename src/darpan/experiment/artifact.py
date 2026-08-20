"""Integrity sealing and verification for durable Darpan research artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from darpan._version import __version__

from .recorder import ResultRecorder

ARTIFACT_MANIFEST = "artifact-manifest.json"
ARTIFACT_MANIFEST_VERSION = 2


def _canonical_hash(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _files(
    directory: Path,
    *,
    include_nested_manifests: bool = True,
) -> list[dict[str, Any]]:
    items = []
    own_manifest = directory / ARTIFACT_MANIFEST
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path == own_manifest:
            continue
        if not include_nested_manifests and path.name == ARTIFACT_MANIFEST:
            continue
        items.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": ResultRecorder.sha256(path),
            }
        )
    return items


def require_fresh_artifact_directory(directory: str | Path) -> Path:
    """Refuse to mix a new research run with stale files from an old run."""

    root = Path(directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"artifact output directory is not empty; choose a fresh path: {root}"
        )
    return root


def seal_artifact(
    directory: str | Path,
    *,
    schema: str,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal every current file in ``directory`` into one integrity manifest."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    files = _files(root)
    payload: dict[str, Any] = {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "schema": schema,
        "darpan_version": __version__,
        "identity": dict(identity or {}),
        "files": files,
        "file_count": len(files),
    }
    payload["content_fingerprint"] = _canonical_hash(files)
    payload["manifest_fingerprint"] = _canonical_hash(
        {
            "manifest_version": payload["manifest_version"],
            "schema": payload["schema"],
            "darpan_version": payload["darpan_version"],
            "identity": payload["identity"],
            "files": files,
        }
    )
    manifest = root / ARTIFACT_MANIFEST
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def restore_artifact_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Restore the last sealed checkpoint, discarding only unsealed extra files.

    Every file recorded by the checkpoint manifest must still match exactly. Files
    created after that checkpoint are considered in-flight evidence and are removed
    so a resumable workflow can restart from the last durable boundary.
    """

    root = Path(directory).expanduser().resolve()
    manifest_path = root / ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"artifact checkpoint manifest does not exist: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    if not isinstance(expected, dict) or not isinstance(expected.get("files"), list):
        raise ValueError("artifact checkpoint manifest is malformed")

    expected_files = expected["files"]
    manifest_version = int(expected.get("manifest_version", 1))
    manifest_basis = {
        "schema": expected.get("schema"),
        "darpan_version": expected.get("darpan_version"),
        "identity": expected.get("identity", {}),
        "files": expected_files,
    }
    if manifest_version >= 2:
        manifest_basis["manifest_version"] = manifest_version
    if _canonical_hash(manifest_basis) != expected.get("manifest_fingerprint"):
        raise RuntimeError("artifact checkpoint manifest fingerprint is invalid")

    expected_map = {str(item["path"]): item for item in expected_files}
    own_manifest = root / ARTIFACT_MANIFEST
    current_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path != own_manifest
        and (manifest_version >= 2 or path.name != ARTIFACT_MANIFEST)
    }
    mismatches = []
    for relative, item in expected_map.items():
        path = current_paths.get(relative)
        observed = None
        if path is not None:
            observed = {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": ResultRecorder.sha256(path),
            }
        if observed != item:
            mismatches.append({"path": relative, "expected": item, "observed": observed})
    if mismatches:
        raise RuntimeError(
            "artifact checkpoint cannot be restored because sealed files changed: "
            f"{len(mismatches)} mismatch(es)"
        )

    extras = sorted(set(current_paths) - set(expected_map))
    for relative in extras:
        current_paths[relative].unlink()
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return {
        "schema": expected.get("schema"),
        "darpan_version": expected.get("darpan_version"),
        "manifest_version": manifest_version,
        "identity": expected.get("identity", {}),
        "restored_files": len(expected_files),
        "discarded_unsealed_files": extras,
        "manifest_fingerprint": expected.get("manifest_fingerprint"),
    }


def verify_artifact(directory: str | Path) -> dict[str, Any]:
    """Verify a sealed artifact exactly, including unexpected extra files."""

    root = Path(directory).expanduser().resolve()
    manifest_path = root / ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    if not isinstance(expected, dict) or not isinstance(expected.get("files"), list):
        raise ValueError("artifact manifest is malformed")
    manifest_version = int(expected.get("manifest_version", 1))
    observed_files = _files(
        root,
        include_nested_manifests=manifest_version >= 2,
    )
    expected_files = expected["files"]
    mismatches = []
    expected_map = {item["path"]: item for item in expected_files}
    observed_map = {item["path"]: item for item in observed_files}
    for path in sorted(set(expected_map) | set(observed_map)):
        before = expected_map.get(path)
        after = observed_map.get(path)
        if before != after:
            mismatches.append({"path": path, "expected": before, "observed": after})
    observed_content = _canonical_hash(observed_files)
    expected_content = expected.get("content_fingerprint")
    manifest_basis = {
        "schema": expected.get("schema"),
        "darpan_version": expected.get("darpan_version"),
        "identity": expected.get("identity", {}),
        "files": expected_files,
    }
    if manifest_version >= 2:
        manifest_basis["manifest_version"] = manifest_version
    expected_manifest = _canonical_hash(manifest_basis)
    manifest_ok = expected_manifest == expected.get("manifest_fingerprint")
    verified = not mismatches and observed_content == expected_content and manifest_ok
    report = {
        "verified": verified,
        "schema": expected.get("schema"),
        "darpan_version": expected.get("darpan_version"),
        "manifest_version": manifest_version,
        "identity": expected.get("identity", {}),
        "file_count": len(observed_files),
        "content_fingerprint": observed_content,
        "manifest_fingerprint": expected.get("manifest_fingerprint"),
        "mismatches": mismatches,
        "manifest_valid": manifest_ok,
    }
    if not verified:
        raise RuntimeError(
            "artifact integrity verification failed: "
            f"{len(mismatches)} file mismatch(es), manifest_valid={manifest_ok}"
        )
    return report
