"""Headless observation configuration and request-bound archive validation."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from r0.modules.priors import (
    CausalLimits,
    ColumnEncoding,
    MeasurementStage,
    MissingnessPlan,
    QuantileColumnSpec,
    TrainingTask,
    column_encodings_from_dict,
    measurement_stages_from_dict,
    observe_missingness,
    quantile_specs_from_dict,
    restore_observation_source,
)


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    plan: MissingnessPlan
    seed: int = 30
    latent: tuple[float, ...] = ()
    latent_provenance: str | None = None
    measurements: tuple[MeasurementStage, ...] = ()
    encodings: tuple[ColumnEncoding, ...] = ()
    quantiles: tuple[QuantileColumnSpec, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.quantiles) is not tuple
            or len(self.quantiles) > 256
            or any(type(s) is not QuantileColumnSpec for s in self.quantiles)
            or len({s.column for s in self.quantiles}) != len(self.quantiles)
        ):
            raise ValueError("quantiles require unique bounded column specifications")
        if (
            type(self.encodings) is not tuple
            or len(self.encodings) > 256
            or any(type(e) is not ColumnEncoding for e in self.encodings)
            or len({e.column for e in self.encodings}) != len(self.encodings)
        ):
            raise ValueError("observation encodings require unique bounded frozen mappings")
        if (
            type(self.measurements) is not tuple
            or len(self.measurements) > 64
            or any(type(s) is not MeasurementStage for s in self.measurements)
        ):
            raise ValueError("observation measurements require bounded frozen stages")
        if (
            type(self.plan) is not MissingnessPlan
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.latent) is not tuple
            or len(self.latent) > 20000
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.latent)
        ):
            raise ValueError("invalid observation request")
        if self.plan.mode == "mnar-latent":
            if (
                not self.latent
                or type(self.latent_provenance) is not str
                or not self.latent_provenance.strip()
            ):
                raise ValueError("latent observation requires a declared row driver and provenance")
        elif self.latent or self.latent_provenance is not None:
            raise ValueError("unused latent observation controls")

    def apply(self, task: TrainingTask, limits: CausalLimits) -> TrainingTask:
        return observe_missingness(
            task,
            self.plan,
            seed=self.seed,
            measurements=self.measurements,
            encodings=self.encodings,
            quantiles=self.quantiles,
            latent=torch.tensor(self.latent, dtype=torch.float64) if self.latent else None,
            latent_provenance=self.latent_provenance,
            limits=limits,
        )

    def matches(self, record: dict[str, Any], limits: CausalLimits) -> bool:
        actual = record.get("observation")
        if type(actual) is not dict:
            return False
        actual = cast(dict[str, Any], actual)
        expected = {
            "version": "r0-missing-observation-v4"
            if self.quantiles
            else "r0-missing-observation-v3"
            if self.encodings
            else "r0-missing-observation-v2"
            if self.measurements
            else "r0-missing-observation-v1",
            "plan": asdict(self.plan),
            "seed": self.seed,
            "limits": asdict(limits),
            "driver_provenance": "original numeric target encoding"
            if self.plan.mode == "mnar-target"
            else self.latent_provenance,
        }
        if bool(actual.get("quantile")) != bool(self.quantiles):
            return False
        if self.quantiles:
            quantile = actual.get("quantile")
            if type(quantile) is not dict or json.dumps(
                cast(dict[str, Any], quantile).get("specs"), sort_keys=True
            ) != json.dumps([asdict(s) for s in self.quantiles], sort_keys=True):
                return False
        if self.encodings:
            encoding = actual.get("encoding")
            if type(encoding) is not dict or json.dumps(
                cast(dict[str, Any], encoding).get("encodings"), sort_keys=True
            ) != json.dumps([asdict(e) for e in self.encodings], sort_keys=True):
                return False
        if bool(actual.get("measurement")) != bool(self.measurements):
            return False
        if self.measurements:
            measurement = actual.get("measurement")
            if type(measurement) is not dict or json.dumps(
                cast(dict[str, Any], measurement).get("stages"), sort_keys=True
            ) != json.dumps([asdict(s) for s in self.measurements], sort_keys=True):
                return False
        return all(
            json.dumps(actual.get(k), sort_keys=True, allow_nan=False)
            == json.dumps(v, sort_keys=True, allow_nan=False)
            for k, v in expected.items()
        )

    def validate(self, task: TrainingTask, limits: CausalLimits) -> TrainingTask:
        if not self.matches(json.loads(task.recipe_json), limits):
            raise ValueError("observation task differs from requested plan")
        source = restore_observation_source(task)
        if self.plan.mode == "mnar-latent":
            assert task.observation_truth is not None
            actual = task.observation_truth.driver
            if actual is None or not torch.equal(
                actual, torch.tensor(self.latent, dtype=torch.float64)
            ):
                raise ValueError("retained latent driver differs from requested row driver")
        return source


def observation_request_from_dict(data: dict[str, Any]) -> ObservationRequest:
    fields: dict[str, Any] = dict(data)
    if "quantiles" in fields:
        value = fields["quantiles"]
        if type(value) not in (list, tuple):
            raise ValueError("quantiles must be an array")
        fields["quantiles"] = quantile_specs_from_dict(value) if value else ()
    if "encodings" in fields:
        value = fields["encodings"]
        if type(value) not in (list, tuple):
            raise ValueError("encodings must be an array")
        fields["encodings"] = column_encodings_from_dict(value) if value else ()
    if "measurements" in fields:
        value = fields["measurements"]
        if type(value) not in (list, tuple):
            raise ValueError("measurements must be an array")
        fields["measurements"] = measurement_stages_from_dict(value) if value else ()
    raw: dict[str, Any] = dict(fields.pop("plan"))
    for key in ("columns", "drivers", "coefficients"):
        if key in raw:
            if type(raw[key]) not in (list, tuple):
                raise ValueError("observation plan arrays required")
            raw[key] = tuple(raw[key])
    if "latent" in fields:
        if type(fields["latent"]) not in (list, tuple):
            raise ValueError("observation latent driver must be an array")
        fields["latent"] = tuple(fields["latent"])
    return ObservationRequest(plan=MissingnessPlan(**raw), **fields)
