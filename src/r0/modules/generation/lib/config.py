"""The same explicit TOML request is used by notebook and hosted entrypoints."""

import tomllib
from pathlib import Path

from r0.modules.generation.lib.requests import RunRequest, request_from_dict


def load_request(path: Path) -> RunRequest:
    config = tomllib.loads(path.read_text())
    if set(config) not in ({"recipe", "run"}, {"recipe", "run", "sampling"}):
        raise ValueError(
            "generation configuration requires recipe/run and optional sampling tables"
        )
    if {"recipe", "sampling"} & set(config["run"]):
        raise ValueError("recipe and sampling belong in their own configuration tables")
    return request_from_dict(
        {**config["run"], "recipe": config["recipe"], "sampling": config.get("sampling")}
    )
