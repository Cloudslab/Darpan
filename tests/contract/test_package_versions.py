from __future__ import annotations

import tomllib
from pathlib import Path

import darpan_rl

import darpan


def _dynamic_version(project: Path, package: Path) -> str:
    data = tomllib.loads(project.read_text(encoding="utf-8"))
    assert data["project"]["dynamic"] == ["version"]
    namespace: dict[str, str] = {}
    exec(package.read_text(encoding="utf-8"), namespace)
    return namespace["__version__"]


def test_core_and_rl_companion_versions_are_kept_in_lockstep():
    root = Path(__file__).parents[2]
    core = _dynamic_version(root / "pyproject.toml", root / "src/darpan/_version.py")
    companion = _dynamic_version(
        root / "packages/darpan-rl/pyproject.toml",
        root / "packages/darpan-rl/src/darpan_rl/_version.py",
    )
    assert core == companion == "0.2.0.dev27"
    assert darpan.__version__ == core
    assert darpan_rl.__version__ == core
