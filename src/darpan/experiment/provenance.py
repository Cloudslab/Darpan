"""Runtime provenance captured with reproducible experiment artifacts."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Provenance:
    python: str
    platform: str
    darpan_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    dependencies: Mapping[str, str] = field(default_factory=dict)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_provenance(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def collect_provenance(*, project_root: str | Path | None = None) -> Provenance:
    darpan_version = _package_version("darpan-continuum") or "source"
    commit, dirty = _git_provenance(
        None if project_root is None else Path(project_root).resolve()
    )
    dependencies = {
        name: item
        for name in ("numpy", "PyYAML")
        if (item := _package_version(name)) is not None
    }
    return Provenance(
        python=sys.version.split()[0],
        platform=platform.platform(),
        darpan_version=darpan_version,
        git_commit=commit,
        git_dirty=dirty,
        dependencies=dependencies,
    )
