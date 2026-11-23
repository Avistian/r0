"""Strict, resolved TOML configuration for the bounded fixture command."""

import tomllib
from dataclasses import dataclass
from typing import cast

from r0.modules.episodes import TaskType


@dataclass(frozen=True)
class Config:
    fixture: TaskType
    version: int = 1
    generator_version: str = "fixture-v1"
    preprocessing: str = "context-v1"
    root_seed: int = 0
    split_seed: int = 0
    task_key: str = "example"
    episode_key: str = "example"
    context_rows: int = 3

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported configuration version; expected 1")
        if self.fixture not in ("regression", "binary", "multiclass"):
            raise ValueError("fixture must be regression, binary or multiclass")
        if self.generator_version != "fixture-v1" or self.preprocessing != "context-v1":
            raise ValueError("unsupported generator or preprocessing version")
        for name, value in (("root_seed", self.root_seed), ("split_seed", self.split_seed)):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0, 2**63)")
        if type(self.context_rows) is not int or not 1 <= self.context_rows <= 5:
            raise ValueError("context_rows must be between 1 and 5 for the six-row fixture")
        for name, value in (("task_key", self.task_key), ("episode_key", self.episode_key)):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


def parse_config(text: str) -> Config:
    """Resolve defaults after validating exact scalar types and known keys."""
    try:
        values: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML: {error}") from error
    defaults: dict[str, object] = {
        "version": 1,
        "fixture": "regression",
        "generator_version": "fixture-v1",
        "preprocessing": "context-v1",
        "root_seed": 0,
        "split_seed": 0,
        "task_key": "example",
        "episode_key": "example",
        "context_rows": 3,
    }
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    if not {"version", "fixture"} <= values.keys():
        raise ValueError("configuration requires version and fixture")
    for key, value in values.items():
        if type(value) is not type(defaults[key]):
            raise ValueError(f"{key}: expected {type(defaults[key]).__name__}")
    defaults.update(values)
    return Config(
        fixture=cast(TaskType, defaults["fixture"]),
        version=cast(int, defaults["version"]),
        generator_version=cast(str, defaults["generator_version"]),
        preprocessing=cast(str, defaults["preprocessing"]),
        root_seed=cast(int, defaults["root_seed"]),
        split_seed=cast(int, defaults["split_seed"]),
        task_key=cast(str, defaults["task_key"]),
        episode_key=cast(str, defaults["episode_key"]),
        context_rows=cast(int, defaults["context_rows"]),
    )
