"""Missingness observation tasks with privileged raw-value and draw custody."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.missingness import MissingnessPlan, draw_missingness
from r0.modules.priors.lib.representatives import encode, record_fingerprint

if TYPE_CHECKING:
    from r0.modules.priors.lib.training_task import TrainingTask

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class ObservationTruth:
    raw_inputs: torch.Tensor
    probabilities: torch.Tensor
    driver: torch.Tensor | None
    record_json: str


def _plan(data: dict[str, Any]) -> MissingnessPlan:
    fields: dict[str, Any] = {
        **data,
        "columns": tuple(data["columns"]),
        "drivers": tuple(data["drivers"]),
        "coefficients": tuple(data["coefficients"]),
    }
    return MissingnessPlan(**fields)


def _rng(seed: int, source: dict[str, Any]) -> np.random.Generator:
    keys = (seed, source["seed"], source["task_id"])
    if any(type(v) is not int or not 0 <= v < 2**63 for v in keys):
        raise ValueError("observation identities require exact nonnegative integers")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F425356, *keys])))


def _identities(record: dict[str, Any]) -> tuple[str, str]:
    source = record_fingerprint((record["source_task_id"], record["plan"], record["seed"]))
    episode = record_fingerprint((record["episode_id"], source, record["binding"]))
    return "observed-" + source, "observed-" + episode


def observe_missingness(
    task: TrainingTask,
    plan: MissingnessPlan,
    *,
    seed: int,
    latent: torch.Tensor | None = None,
    latent_provenance: str | None = None,
    limits: CausalLimits = _DEFAULT,
) -> TrainingTask:
    """Apply one declared mask stage; retain all source values outside predictor arrays.

    Target-driven masks use the original numeric target encoding. A latent driver
    needs an explicit provenance label supplied by its caller. This first wrapper
    rejects already-masked tasks; ordered multi-stage observation is separate.
    """
    if task.missing_mask is not None or task.observation_truth is not None:
        raise ValueError("missingness observation requires a complete unmasked source task")
    if task.inputs.device.type != "cpu" or task.inputs.dtype != torch.float64:
        raise ValueError("missingness observation currently requires CPU float64 inputs")
    if plan.mode == "mnar-latent":
        if (
            latent is None
            or latent.device.type != "cpu"
            or latent.dtype != torch.float64
            or type(latent_provenance) is not str
            or not latent_provenance.strip()
        ):
            raise ValueError("latent mask requires a CPU float64 driver and explicit provenance")
    elif latent is not None or latent_provenance is not None:
        raise ValueError("unused latent mask driver or provenance")
    source = json.loads(task.recipe_json)
    if "observation" in source:
        raise ValueError("source already carries observation metadata")
    driver = task.targets.to(torch.float64).clone() if plan.mode == "mnar-target" else latent
    raw = np.asarray(task.inputs.detach().numpy(), dtype=np.float64)
    values = None if driver is None else np.asarray(driver.detach().numpy(), dtype=np.float64)
    sampled = draw_missingness(
        plan,
        raw,
        rng=_rng(seed, source),
        limits=limits,
        target=values if plan.mode == "mnar-target" else None,
        latent=values if plan.mode == "mnar-latent" else None,
    )
    record = {
        "version": "r0-missing-observation-v1",
        "plan": asdict(plan),
        "seed": seed,
        "binding": sampled.binding,
        "initial": asdict(sampled.initial),
        "final": asdict(sampled.final),
        "limits": asdict(limits),
        "source_task_id": task.source_task_id,
        "episode_id": task.episode_id,
        "driver_provenance": "original numeric target encoding"
        if plan.mode == "mnar-target"
        else latent_provenance,
    }
    truth = ObservationTruth(
        task.inputs.clone(),
        torch.tensor(sampled.probabilities),
        None if driver is None else driver.clone(),
        encode(record),
    )
    mask = torch.tensor(sampled.mask)
    inputs = task.inputs.clone()
    inputs[mask] = float("nan")
    source_id, episode_id = _identities(record)
    return replace(
        task,
        inputs=inputs,
        missing_mask=mask,
        observation_truth=truth,
        source_task_id=source_id,
        episode_id=episode_id,
        recipe_json=encode({**source, "observation": record}),
    )


def validate_missing_observation(task: TrainingTask) -> None:
    """Check stored inputs, probabilities, drivers, identities and draw replay.

    This verifies the observation layer against retained source values. Re-running
    the underlying generator is a separate source-replay check.
    """
    truth = task.observation_truth
    if type(truth) is not ObservationTruth or task.missing_mask is None:
        raise ValueError("observation task requires privileged truth and explicit mask")
    raw = truth.raw_inputs
    if (
        raw.device.type != "cpu"
        or raw.dtype != torch.float64
        or raw.shape != task.inputs.shape
        or task.inputs.device.type != "cpu"
        or truth.probabilities.dtype != torch.float64
        or truth.probabilities.shape != raw.shape
        or truth.probabilities.device.type != "cpu"
    ):
        raise ValueError("observation truth geometry or device differs")
    record = json.loads(truth.record_json)
    expected_fields = {
        "version",
        "plan",
        "seed",
        "binding",
        "initial",
        "final",
        "limits",
        "source_task_id",
        "episode_id",
        "driver_provenance",
    }
    source = json.loads(task.recipe_json)
    if (
        set(record) != expected_fields
        or record["version"] != "r0-missing-observation-v1"
        or encode(source.get("observation")) != encode(record)
        or _identities(record) != (task.source_task_id, task.episode_id)
    ):
        raise ValueError("observation metadata or identity differs")
    plan = _plan(record["plan"])
    driver = truth.driver
    if plan.mode == "mnar-target":
        if (
            driver is None
            or not torch.equal(driver, task.targets.to(torch.float64))
            or record["driver_provenance"] != "original numeric target encoding"
        ):
            raise ValueError("target mask driver differs from retained targets")
    elif plan.mode == "mnar-latent":
        if type(record["driver_provenance"]) is not str or not record["driver_provenance"].strip():
            raise ValueError("latent mask driver needs provenance")
    elif driver is not None or record["driver_provenance"] is not None:
        raise ValueError("unexpected privileged observation driver")
    if driver is not None and (driver.device.type != "cpu" or driver.dtype != torch.float64):
        raise ValueError("observation driver must be CPU float64")
    values = None if driver is None else np.asarray(driver.detach().numpy(), dtype=np.float64)
    sampled = draw_missingness(
        plan,
        np.asarray(raw.detach().numpy(), dtype=np.float64),
        rng=_rng(record["seed"], source),
        limits=CausalLimits(**record["limits"]),
        target=values if plan.mode == "mnar-target" else None,
        latent=values if plan.mode == "mnar-latent" else None,
    )
    if (
        sampled.binding != record["binding"]
        or encode(asdict(sampled.initial)) != encode(record["initial"])
        or encode(asdict(sampled.final)) != encode(record["final"])
        or not torch.equal(torch.tensor(sampled.mask), task.missing_mask)
        or not torch.equal(torch.tensor(sampled.probabilities), truth.probabilities)
        or not torch.equal(task.inputs[~task.missing_mask], raw[~task.missing_mask])
    ):
        raise ValueError("observation values or mask differ from retained replay")


def restore_observation_source(task: TrainingTask) -> TrainingTask:
    """Restore retained source buffers for separate structural replay/inspection."""
    validate_missing_observation(task)
    truth = task.observation_truth
    assert truth is not None
    record = json.loads(truth.record_json)
    source = json.loads(task.recipe_json)
    source.pop("observation")
    return replace(
        task,
        inputs=truth.raw_inputs.clone(),
        missing_mask=None,
        observation_truth=None,
        source_task_id=record["source_task_id"],
        episode_id=record["episode_id"],
        recipe_json=encode(source),
    )
