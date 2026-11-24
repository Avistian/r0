"""Canonical JSON and integrity-checked, append-only local bundles."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

Document = dict[str, Any]


def encoded(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(encoded(value) + b"\n")


def read_json(path: Path) -> Document:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(Document, value)


def seal(root: Path) -> str:
    files = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "integrity.json"
    }
    receipt = {"version": 1, "files": files}
    write_json(root / "integrity.json", receipt)
    return digest(receipt)


def verify(root: Path, expected: str | None = None) -> None:
    receipt = read_json(root / "integrity.json")
    if expected is not None and digest(receipt) != expected:
        raise ValueError("bundle integrity receipt mismatch")
    actual = {
        str(p.relative_to(root)): file_hash(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "integrity.json"
    }
    if actual != receipt.get("files") or receipt.get("version") != 1:
        raise ValueError("bundle corruption; preserve this bundle and use a fresh attempt")
