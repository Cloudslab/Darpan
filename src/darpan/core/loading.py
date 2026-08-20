"""Explicit loading for researcher-owned Python plugins.

Plugins may be importable package paths (``package.module:Symbol``) or Python
files next to an experiment configuration (``my_policy.py:Policy``).  Local
files are loaded without requiring the research project to be installed as a
Python package.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _split_reference(path: str) -> tuple[str, str]:
    if ":" not in path:
        raise ValueError("plugin path must use 'module:Symbol' syntax")
    module_name, symbol_name = path.rsplit(":", 1)
    if not module_name or not symbol_name:
        raise ValueError("invalid plugin path")
    return module_name, symbol_name


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted and value in sys.path:
            sys.path.remove(value)


def _local_candidate(module_name: str, search_path: Path) -> Path | None:
    raw = Path(module_name)
    if raw.suffix == ".py":
        candidate = raw if raw.is_absolute() else search_path / raw
    else:
        candidate = search_path.joinpath(*module_name.split(".")).with_suffix(".py")
    candidate = candidate.resolve()
    return candidate if candidate.is_file() else None


def _load_local_module(module_name: str, file_path: Path):
    digest = hashlib.sha256(str(file_path).encode()).hexdigest()[:12]
    internal_name = f"_darpan_plugin_{file_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(internal_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[internal_name] = module
    try:
        with _temporary_sys_path(file_path.parent):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(internal_name, None)
        raise
    return module



def resolve_local_plugin_file(
    path: str, *, search_path: str | Path | None = None
) -> Path | None:
    """Resolve a researcher-owned plugin reference without importing it.

    This mirrors :func:`load_symbol` local-file precedence so reproducibility
    tooling can hash the exact policy/metric/model source used by an
    experiment. Installed package plugins intentionally return ``None`` and
    remain represented by package provenance instead.
    """

    module_name, _ = _split_reference(path)
    if search_path is not None:
        candidate = _local_candidate(module_name, Path(search_path).resolve())
        if candidate is not None:
            return candidate
    if module_name.endswith(".py"):
        candidate = Path(module_name).expanduser()
        if not candidate.is_absolute() and search_path is not None:
            candidate = Path(search_path).resolve() / candidate
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None
    return None

def load_symbol(path: str, *, search_path: str | Path | None = None) -> Any:
    """Load a plugin symbol without modifying Darpan source.

    When ``search_path`` is supplied, a matching local Python file takes
    precedence over installed modules.  This is what lets researchers keep
    ``my_policy.py`` beside ``experiment.yaml`` and run it directly.
    """

    module_name, symbol_name = _split_reference(path)
    module = None
    if search_path is not None:
        candidate = _local_candidate(module_name, Path(search_path).resolve())
        if candidate is not None:
            module = _load_local_module(module_name, candidate)
    if module is None:
        if module_name.endswith(".py"):
            candidate = Path(module_name).resolve()
            if not candidate.is_file():
                raise ImportError(f"plugin file does not exist: {module_name}")
            module = _load_local_module(module_name, candidate)
        else:
            module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(f"{symbol_name!r} not found in {module_name!r}") from exc


def instantiate(
    path: str,
    *args,
    search_path: str | Path | None = None,
    **kwargs,
) -> Any:
    symbol = load_symbol(path, search_path=search_path)
    return symbol(*args, **kwargs)


def materialize_plugin(path: str, *, search_path: str | Path | None = None) -> Any:
    """Return an exported plugin instance, instantiating classes with no args."""

    symbol = load_symbol(path, search_path=search_path)
    return symbol() if inspect.isclass(symbol) else symbol
