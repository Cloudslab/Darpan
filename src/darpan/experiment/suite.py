"""Frozen paper-suite manifests for stable experiment identities and inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import resolved_inputs
from .spec import ExperimentSpec

SUPPORTED_SUITE_ITEM_KINDS = frozenset(
    {
        "application",
        "cluster",
        "evidence",
        "experiment",
        "scenario",
        "system",
        "workload",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()



@dataclass(frozen=True, slots=True)
class SuiteItem:
    id: str
    kind: str
    path: str
    role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("suite item id cannot be empty")
        if self.kind not in SUPPORTED_SUITE_ITEM_KINDS:
            allowed = ", ".join(sorted(SUPPORTED_SUITE_ITEM_KINDS))
            raise ValueError(f"unsupported suite item kind {self.kind!r}; choose {allowed}")
        if not self.path:
            raise ValueError(f"suite item {self.id!r} path cannot be empty")


@dataclass(frozen=True, slots=True)
class PaperSuite:
    name: str
    version: str
    items: tuple[SuiteItem, ...]
    required_evidence: tuple[str, ...] = ()
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("suite name cannot be empty")
        if not self.version:
            raise ValueError("suite version cannot be empty")
        if not self.items:
            raise ValueError("suite requires at least one item")
        ids = tuple(item.id for item in self.items)
        if len(set(ids)) != len(ids):
            raise ValueError("suite item ids must be unique")
        if any(not item for item in self.required_evidence):
            raise ValueError("suite required_evidence entries cannot be empty")
        if len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("suite required_evidence entries must be unique")

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def resolve_path(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.directory / raw).resolve()

    def item(self, item_id: str) -> SuiteItem:
        if item_id.startswith("suite:"):
            clean = item_id.removeprefix("suite:")
        else:
            clean = item_id[1:] if item_id.startswith("@") else item_id
        try:
            return next(item for item in self.items if item.id == clean)
        except StopIteration as exc:
            raise KeyError(f"suite {self.name!r} has no item {clean!r}") from exc

    def resolve_reference(
        self,
        value: str,
        *,
        kinds: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> Path:
        if not (value.startswith("@") or value.startswith("suite:")):
            return self.resolve_path(value)
        item = self.item(value)
        if kinds is not None and item.kind not in kinds:
            allowed = ", ".join(sorted(kinds))
            raise ValueError(
                f"suite item {item.id!r} has kind {item.kind!r}; expected {allowed}"
            )
        return self.resolve_path(item.path)

    def _item_inputs(self, item: SuiteItem) -> tuple[Path, ...]:
        source = self.resolve_path(item.path)
        if not source.is_file():
            raise FileNotFoundError(f"suite item does not exist: {source}")
        if item.kind != "experiment":
            return (source,)
        spec = ExperimentSpec.load(source)
        return tuple(resolved_inputs(spec))

    def lock_payload(self) -> dict[str, Any]:
        base = self.directory
        locked_items: dict[str, Any] = {}
        for item in sorted(self.items, key=lambda value: value.id):
            inputs = []
            for source in sorted(self._item_inputs(item), key=lambda value: str(value)):
                inputs.append(
                    {
                        "path": Path(
                            os.path.relpath(source.resolve(), base.resolve())
                        ).as_posix(),
                        "sha256": _sha256(source),
                    }
                )
            body = {
                "kind": item.kind,
                "role": item.role,
                "path": item.path,
                "metadata": dict(item.metadata),
                "inputs": inputs,
            }
            body["fingerprint"] = _canonical_hash(body)
            locked_items[item.id] = body
        payload: dict[str, Any] = {
            "schema": "darpan.paper-suite.lock/v1",
            "name": self.name,
            "version": self.version,
            "required_evidence": list(self.required_evidence),
            "items": locked_items,
        }
        payload["fingerprint"] = _canonical_hash(payload)
        return payload

    def verify_lock(self, lock_path: str | Path) -> dict[str, Any]:
        path = Path(lock_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
        actual = self.lock_payload()
        if expected != actual:
            expected_fp = expected.get("fingerprint") if isinstance(expected, dict) else None
            raise RuntimeError(
                "paper suite lock mismatch: "
                f"expected {expected_fp!r}, observed {actual['fingerprint']!r}"
            )
        return actual

    @classmethod
    def load(cls, path: str | Path) -> PaperSuite:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("suite configuration must be a mapping")
        raw_items = data.get("items", ())
        if not isinstance(raw_items, list):
            raise ValueError("suite items must be a list")
        items = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError("each suite item must be a mapping")
            items.append(
                SuiteItem(
                    id=str(raw["id"]),
                    kind=str(raw["kind"]),
                    path=str(raw["path"]),
                    role=None if raw.get("role") is None else str(raw["role"]),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        suite = cls(
            name=str(data.get("name", source.stem)),
            version=str(data.get("version", "1")),
            items=tuple(items),
            required_evidence=tuple(
                str(item) for item in data.get("required_evidence", ())
            ),
            source=source,
        )
        for item in suite.items:
            suite._item_inputs(item)
        return suite
