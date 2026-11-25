"""The same explicit TOML request is used by notebook and hosted entrypoints."""

import tomllib
from pathlib import Path

from r0.modules.generation.lib.runs import RunRequest
from r0.modules.priors import recipe_from_dict


def load_request(path: Path) -> RunRequest:
    config = tomllib.loads(path.read_text())
    if set(config) != {"recipe", "run"}:
        raise ValueError("generation configuration requires only recipe and run tables")
    return RunRequest(recipe=recipe_from_dict(config["recipe"]), **config["run"])
