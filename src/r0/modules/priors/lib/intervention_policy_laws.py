"""Explicit numerical parameter laws; never infer a prior from one frozen function."""

import math
from dataclasses import dataclass, is_dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class InterventionParameterLaw:
    node: int
    path: tuple[str, ...]
    family: str = "normal-r0"
    location: float = 0.0
    scale: float = 1.0
    choices: tuple[float, ...] = ()
    version: str = "intervention-parameter-law-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "intervention-parameter-law-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.path) is not tuple
            or not self.path
            or any(type(x) is not str or not x or len(x) > 128 for x in self.path)
            or self.family not in ("normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0")
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.location, self.scale)
            )
            or self.scale <= 0
            or type(self.choices) is not tuple
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.choices)
            or (self.family == "choice-r0") != bool(self.choices)
        ):
            raise ValueError("parameter laws require explicit finite domains and frozen paths")
        if self.family == "log-uniform-r0" and (self.location <= 0 or self.scale <= self.location):
            raise ValueError("log-uniform parameter bounds must be positive and increasing")


def parameter_group(record: Any, path: tuple[str, ...]) -> tuple[Any, str | int, Any]:
    def key_for(parent: Any, token: str) -> str | int:
        if type(parent) not in (tuple, list):
            return token
        if not token.isdecimal() or str(int(token)) != token or not 0 <= int(token) < len(parent):
            raise ValueError("parameter sequence indices must be canonical nonnegative decimals")
        return int(token)

    def check_custody(parent: Any, token: str) -> None:
        version = (
            getattr(parent, "version", None)
            if is_dataclass(parent)
            else (cast(dict[str, Any], parent).get("version") if type(parent) is dict else None)
        )
        if version == "sampled-feature-conv1d-r0-v1" and token != "function":
            raise ValueError("policy laws cannot alter sampled convolution custody")

    parent: Any = record
    for token in path[:-1]:
        check_custody(parent, token)
        key = key_for(parent, token)
        parent = getattr(parent, str(key)) if is_dataclass(parent) else parent[key]
    check_custody(parent, path[-1])
    key = key_for(parent, path[-1])
    return parent, key, getattr(parent, str(key)) if is_dataclass(parent) else parent[key]


def numeric_leaves(value: Any) -> int:
    if type(value) is float and math.isfinite(value):
        return 1
    if type(value) in (tuple, list) and value:
        return sum(numeric_leaves(v) for v in cast(tuple[Any, ...] | list[Any], value))
    raise ValueError(
        "resampled groups must contain finite float parameters, never indices or flags"
    )


def replace_leaves(value: Any, values: Any) -> Any:
    if type(value) is float:
        return float(next(values))
    return [replace_leaves(v, values) for v in value]
