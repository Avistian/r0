"""Headless observation configuration and request-bound archive validation."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from r0.modules.priors import (
    CausalLimits,
    MissingnessPlan,
    TrainingTask,
    observe_missingness,
    restore_observation_source,
)


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    plan: MissingnessPlan
    seed: int = 30
    latent: tuple[float, ...] = ()
    latent_provenance: str | None = None

    def __post_init__(self) -> None:
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
            "version": "r0-missing-observation-v1",
            "plan": asdict(self.plan),
            "seed": self.seed,
            "limits": asdict(limits),
            "driver_provenance": "original numeric target encoding"
            if self.plan.mode == "mnar-target"
            else self.latent_provenance,
        }
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
