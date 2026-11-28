"""Immutable original-row causal reference receipts for retained finite GP-root DAGs."""

from dataclasses import asdict, dataclass
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_population_metadata_cost import retained_cost
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.training_features import TrainingFeature

_DEFAULT_LIMITS = CausalLimits()

type GPReferenceSource = GPRootDAGRecipe | GPRootCalibration | GPRootCalibrationQuery


@dataclass(frozen=True, slots=True)
class GPRootReference:
    source_json: str
    source_fingerprint: str
    population_id: str
    fit_population_id: str | None
    seed: int
    task_id: int
    row: int
    inputs: tuple[tuple[float, ...], ...]
    feature_schema: tuple[TrainingFeature, ...]
    structural_innovations: str
    capture_location: str
    raw_scores: tuple[tuple[float, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    target_mean: tuple[float, ...] | None
    probabilities: tuple[tuple[float, ...], ...] | None
    targets: tuple[float | int, ...]
    classes: tuple[str, ...]
    version: str = "gp-root-reference-v1"
    semantics: str = "observed-clamps-original-hidden-population-not-conditional-mean-v1"
    target_draws: str = "fixed-original-row-task-adapter-and-domain-draws-v1"


def evaluate_gp_root_reference(
    source: GPReferenceSource,
    *,
    row: int,
    inputs: tuple[tuple[float, ...], ...],
    structural_innovations: str = "original-row",
    capture_location: str = "capture",
    limits: CausalLimits | None = None,
) -> GPRootReference:
    if type(source) not in (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery):
        raise ValueError("GP reference requires a complete retained GP-root recipe")
    cap = source.limits if limits is None else limits
    if type(cap) is not CausalLimits:
        raise ValueError("GP reference requires explicit caller limits")
    retained_cost(source, cap)
    return gp_root_reference_from_source(
        asdict(source),
        row=row,
        inputs=inputs,
        structural_innovations=structural_innovations,
        capture_location=capture_location,
        limits=cap,
    )


def gp_root_reference_from_source(
    source: Any,
    *,
    row: int,
    inputs: tuple[tuple[float, ...], ...],
    structural_innovations: str = "original-row",
    capture_location: str = "capture",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPRootReference:
    from r0.modules.priors.lib.gp_reference_admission import admit_gp_reference
    from r0.modules.priors.lib.gp_reference_execution import realize_reference

    plan = admit_gp_reference(
        source,
        row=row,
        inputs=inputs,
        structural_innovations=structural_innovations,
        capture_location=capture_location,
        limits=limits,
    )
    return realize_reference(
        source,
        plan,
        row=row,
        inputs=inputs,
        structural_innovations=structural_innovations,
        capture_location=capture_location,
    )
