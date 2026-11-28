"""The same explicit TOML request is used by notebook and hosted entrypoints."""

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from r0.modules.generation.lib.requests import RunRequest, request_from_dict


def load_request(path: Path) -> RunRequest:
    return request_from_dict(load_request_document(path))


def load_request_document(path: Path) -> dict[str, Any]:
    """Read configuration without replaying mechanisms outside a guarded worker.

    JSON accepts the public RunRequest document; TOML uses recipe/run tables.
    """
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text())
        if not isinstance(document, dict):
            raise ValueError("generation request must be an object")
        return cast(dict[str, Any], document)
    config = tomllib.loads(path.read_text())
    if set(config) not in ({"recipe", "run"}, {"recipe", "run", "sampling"}):
        raise ValueError(
            "generation configuration requires recipe/run and optional sampling tables"
        )
    if not isinstance(config["run"], dict):
        raise ValueError("run must be a configuration table")
    run = cast(dict[str, Any], config["run"])
    if {"recipe", "sampling"} & set(run):
        raise ValueError("recipe and sampling belong in their own configuration tables")
    return {**run, "recipe": config["recipe"], "sampling": config.get("sampling")}
