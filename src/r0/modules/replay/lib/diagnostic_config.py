"""Resolved, capped proposal schedules independent of generation and filtering."""

import tomllib
from dataclasses import dataclass
from typing import Literal, cast

from r0.modules.episodes import TaskType
from r0.modules.priors import FAMILIES, Proposal


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    version: int = 1
    families: tuple[str, ...] = FAMILIES
    seeds: tuple[int, ...] = (11, 23, 47)
    draws: int = 2
    rows: int = 64
    features: int = 8
    context_rows: int = 48
    selection: Literal["none", "drop_added_noise"] = "none"
    profile: str = "diagnostic-v1"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1 or self.profile != "diagnostic-v1":
            raise ValueError("unsupported diagnostic configuration version/profile")
        if (
            type(self.families) is not tuple
            or not self.families
            or any(f not in FAMILIES for f in self.families)
        ):
            raise ValueError("families must be a nonempty tuple of supported representatives")
        if len(set(self.families)) != len(self.families):
            raise ValueError("duplicate families are not a weighting policy")
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be unique nonnegative integers below 2**63")
        if (
            type(self.draws) is not int
            or self.draws < 1
            or len(self.families) * len(self.seeds) * self.draws * 2 > 256
        ):
            raise ValueError("diagnostic proposal limit: 1..256")
        if self.selection not in ("none", "drop_added_noise"):
            raise ValueError("unsupported selection policy")
        Proposal("linear", rows=self.rows, features=self.features, context_rows=self.context_rows)
        if "interaction" in self.families and self.features < 3:
            raise ValueError("interaction representative requires at least three features")


def diagnostic_schedule(config: DiagnosticConfig) -> tuple[Proposal, ...]:
    types: tuple[TaskType, ...] = ("regression", "binary", "multiclass")
    return tuple(
        Proposal(
            cast(Literal["linear", "gp", "mlp", "scm", "tree", "interaction"], family),
            seed=seed,
            draw=draw,
            noise=noise,
            task_type=types[(i + draw) % 3],
            rows=config.rows,
            features=config.features,
            context_rows=config.context_rows,
        )
        for family in config.families
        for i, seed in enumerate(config.seeds)
        for draw in range(config.draws)
        for noise in ("none", "added")
    )


def parse_diagnostic_config(text: str) -> DiagnosticConfig:
    values: dict[str, object] = tomllib.loads(text)
    allowed = {
        "version",
        "families",
        "seeds",
        "draws",
        "rows",
        "features",
        "context_rows",
        "selection",
        "profile",
    }
    if values.keys() - allowed:
        raise ValueError(
            f"unknown diagnostic configuration keys: {sorted(values.keys() - allowed)}"
        )
    if "version" not in values:
        raise ValueError("configuration requires version")
    for key in ("families", "seeds"):
        if key in values:
            if not isinstance(values[key], list):
                raise ValueError(f"{key} must be an array")
            values[key] = tuple(cast(list[object], values[key]))
    return DiagnosticConfig(
        version=cast(int, values["version"]),
        families=cast(tuple[str, ...], values.get("families", FAMILIES)),
        seeds=cast(tuple[int, ...], values.get("seeds", (11, 23, 47))),
        draws=cast(int, values.get("draws", 2)),
        rows=cast(int, values.get("rows", 64)),
        features=cast(int, values.get("features", 8)),
        context_rows=cast(int, values.get("context_rows", 48)),
        selection=cast(Literal["none", "drop_added_noise"], values.get("selection", "none")),
        profile=cast(str, values.get("profile", "diagnostic-v1")),
    )
