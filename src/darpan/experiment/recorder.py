"""Small durable writer for paper/research experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ResultRecorder:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str, sort_keys=True)
            handle.write("\n")
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_jsonl(self, name: str, payloads: Iterable[Any]) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, default=str, sort_keys=True))
                handle.write("\n")
        return path

    def copy(self, source: str | Path, name: str | None = None) -> Path:
        source = Path(source)
        destination = self.directory / (name or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    @staticmethod
    def sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
