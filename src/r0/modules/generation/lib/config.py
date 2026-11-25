"""The same explicit TOML request is used by notebook and hosted entrypoints."""

import tomllib
from pathlib import Path

from r0.modules.generation.lib.runs import RunRequest
from r0.modules.priors import LinearRecipe


def load_request(path: Path) -> RunRequest:
    config = tomllib.loads(path.read_text())
    if set(config) != {"recipe", "run"}:
        raise ValueError("generation configuration requires only recipe and run tables")
    recipe = config["recipe"]
    if "class_offsets" in recipe:
        recipe["class_offsets"] = tuple(recipe["class_offsets"])
    return RunRequest(recipe=LinearRecipe(**recipe), **config["run"])
