"""Strict configuration for explicitly authored oracle cards, not sampled generators."""

import math
import tomllib
from dataclasses import dataclass
from typing import Literal, cast

from r0.modules.episodes import Target
from r0.modules.oracles import ClassificationSpec, GPSpec


@dataclass(frozen=True, slots=True)
class OracleConfig:
    name: str
    mode: Literal["prior", "posterior"]
    reference: ClassificationSpec | GPSpec
    context_x: tuple[float, ...]
    context_y: tuple[Target, ...]
    query_x: tuple[float, ...]
    query_targets: tuple[Target, ...]
    feature_unit: str
    target_unit: str
    atol: float = 1e-10
    rtol: float = 1e-10
    tolerance_reason: str = "default well-conditioned reference"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported oracle configuration version")
        if self.mode not in ("prior", "posterior"):
            raise ValueError("mode must be prior or posterior")
        for value in (self.name, self.feature_unit, self.target_unit, self.tolerance_reason):
            if type(value) is not str or not value.strip():
                raise ValueError("name, units and tolerance reason must be nonempty strings")
        if type(self.reference) not in (GPSpec, ClassificationSpec):
            raise ValueError("unsupported reference specification")
        for values in (self.context_x, self.context_y, self.query_x, self.query_targets):
            if type(values) is not tuple:
                raise ValueError("observations must be immutable tuples")
        if len(self.context_x) > 64 or not 1 <= len(self.query_x) <= 64:
            raise ValueError("reference limit: 64 context rows and 1..64 query rows")
        if len(self.context_x) != len(self.context_y):
            raise ValueError("every context coordinate needs a target")
        for x in self.context_x + self.query_x:
            if type(x) not in (float, int) or not math.isfinite(x):
                raise ValueError("coordinates must be finite numbers")
        if self.mode == "prior":
            if isinstance(self.reference, GPSpec) and self.reference.jitter:
                raise ValueError("prior mode has no conditioning matrix for jitter")
            if self.context_x or self.context_y or self.query_targets:
                raise ValueError("prior mode forbids context observations and scoring targets")
        elif not self.context_x or len(self.query_targets) != len(self.query_x):
            raise ValueError("posterior mode requires context and all query scoring targets")
        for target in self.context_y + self.query_targets:
            if isinstance(self.reference, GPSpec):
                if type(target) not in (float, int) or not math.isfinite(float(target)):
                    raise ValueError("GP targets must be finite numbers")
            elif type(target) is not str or target not in self.reference.classes:
                raise ValueError("classification target outside declared vocabulary")
        for value in (self.atol, self.rtol):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite nonnegative numbers")
        if (self.atol, self.rtol) != (1e-10, 1e-10) and (
            self.tolerance_reason == "default well-conditioned reference"
        ):
            raise ValueError("nondefault tolerances require a prior justification")


def _table(value: object, allowed: set[str], required: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    data = cast(dict[str, object], value)
    if data.keys() - allowed:
        raise ValueError(f"unknown {name} keys: {sorted(data.keys() - allowed)}")
    if required - data.keys():
        raise ValueError(f"missing {name} keys: {sorted(required - data.keys())}")
    return data


def _array(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return tuple(cast(list[object], value))


def parse_oracle_config(text: str) -> OracleConfig:
    root = _table(
        tomllib.loads(text),
        {"version", "name", "mode", "reference", "observations", "scoring", "units", "numerics"},
        {"version", "name", "mode", "reference", "observations"},
        "oracle configuration",
    )
    value = root["reference"]
    if not isinstance(value, dict):
        raise ValueError("reference must be a table")
    value = cast(dict[str, object], value)
    family = value.get("family")
    reference: ClassificationSpec | GPSpec
    if family in ("beta_bernoulli", "dirichlet_categorical"):
        table = _table(
            value,
            {"family", "classes", "concentration", "positive_class"},
            {"family", "classes", "concentration"},
            "classification reference",
        )
        reference = ClassificationSpec(
            family,
            cast(tuple[str, ...], _array(table["classes"], "classes")),
            cast(tuple[float, ...], _array(table["concentration"], "concentration")),
            cast(str | None, table.get("positive_class")),
        )
    elif family in ("linear", "rbf"):
        table = _table(
            value,
            {"family", "mean", "signal_variance", "noise_variance", "length_scale", "jitter"},
            {"family"},
            "GP reference",
        )
        reference = GPSpec(
            family,
            cast(float, table.get("mean", 0.0)),
            cast(float, table.get("signal_variance", 1.0)),
            cast(float, table.get("noise_variance", 0.0)),
            cast(float | None, table.get("length_scale")),
            cast(float, table.get("jitter", 0.0)),
        )
    else:
        raise ValueError("unsupported reference family")
    observations = _table(
        root["observations"], {"context_x", "context_y", "query_x"}, {"query_x"}, "observations"
    )
    scoring = _table(root.get("scoring", {}), {"targets"}, set(), "scoring")
    units = _table(
        root.get("units", {}),
        {"feature", "target"},
        {"feature", "target"} if isinstance(reference, GPSpec) else set(),
        "units",
    )
    numerics = _table(root.get("numerics", {}), {"atol", "rtol", "reason"}, set(), "numerics")
    if root["mode"] == "prior" and "scoring" in root:
        raise ValueError("prior mode forbids a scoring section")
    return OracleConfig(
        cast(str, root["name"]),
        cast(Literal["prior", "posterior"], root["mode"]),
        reference,
        cast(tuple[float, ...], _array(observations.get("context_x", []), "context_x")),
        cast(tuple[Target, ...], _array(observations.get("context_y", []), "context_y")),
        cast(tuple[float, ...], _array(observations["query_x"], "query_x")),
        cast(tuple[Target, ...], _array(scoring.get("targets", []), "targets")),
        cast(str, units.get("feature", "unused")),
        cast(str, units.get("target", "class label")),
        cast(float, numerics.get("atol", 1e-10)),
        cast(float, numerics.get("rtol", 1e-10)),
        cast(str, numerics.get("reason", "default well-conditioned reference")),
        cast(int, root["version"]),
    )
