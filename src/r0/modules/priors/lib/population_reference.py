"""Versioned original-row population responses retain full mechanism/source identity."""

from dataclasses import asdict, dataclass
from typing import Any, cast

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
)
from r0.modules.priors.lib.calibrated_population_query import CalibratedPopulationQuery
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.population_reference_admission import admit_population_reference
from r0.modules.priors.lib.population_reference_execution import (
    PopulationReferenceSource,
    evaluate_reference_point,
    overlay_reference,
    reference_clamps,
)
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_decoding import complete_population_record
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.representatives import encode, record_fingerprint

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class PopulationGraphReference:
    source_json: str
    source_fingerprint: str
    seed: int
    task_id: int
    row: int
    feature: int
    values: tuple[float, ...]
    anchor: tuple[float, ...]
    vocabulary: tuple[str, ...]
    target_classes: tuple[str, ...]
    capture_location: str
    scores: tuple[tuple[float, ...], ...]
    probabilities: tuple[tuple[float, ...], ...] | None
    targets: tuple[float | int, ...]
    population_mode: str
    population_index: int | None
    precedence: str = "observed-location-reference-overrides-only-prior-hard-cut-v1"
    semantics: str = "original-row-owned-mechanisms-and-draws-not-conditional-mean-v1"
    version: str = "population-graph-reference-v1"


def _reference_from_source(
    source: Any,
    *,
    row: int,
    feature: int,
    values: tuple[float, ...],
    capture_location: str = "capture",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationGraphReference:
    """Strict raw source entrypoint; all query controls precede nested source replay."""
    from r0.modules.priors.lib.generate import recipe_from_dict

    _, nested, schema, _ = admit_population_reference(
        source,
        row=row,
        feature=feature,
        values=values,
        capture_location=capture_location,
        limits=limits,
    )
    from r0.modules.priors.lib.calibrated_population_reference import (
        CALIBRATED_REFERENCE_TYPES,
        CalibratedPopulationReference,
        calibrated_reference_outputs,
    )

    decoded = recipe_from_dict(source, limits=nested)
    wrapper = None
    if isinstance(decoded, CALIBRATED_REFERENCE_TYPES):
        wrapper = decoded
        decoded = decoded.original
    if not isinstance(decoded, (RecursivePopulationGraph, RecursivePopulationGraphIntervention)):
        raise ValueError("population reference requires its retained population recipe")
    original = decoded.replay(limits=nested).task
    anchor = tuple(float(v) for v in original.inputs[row])
    prior = decoded.change if isinstance(decoded, RecursivePopulationGraphIntervention) else None
    base = decoded.base if isinstance(decoded, RecursivePopulationGraphIntervention) else decoded
    runtime = prior.effective_runtime() if prior else base.request.base
    scores: list[tuple[float, ...]] = []
    probabilities: list[tuple[float, ...]] | None = (
        [] if original.probabilities is not None else None
    )
    targets: list[float | int] = []
    raw_scores: list[tuple[float, ...]] = []
    for value in values:
        point = tuple(value if i == feature else v for i, v in enumerate(anchor))
        change = overlay_reference(
            runtime, prior, reference_clamps(runtime, point, capture_location)
        )
        result = evaluate_reference_point(decoded, runtime, change)
        if wrapper is not None:
            raw_scores.append(tuple(float(v) for v in result.scores[row]))
            point_scores, point_targets, point_probabilities, _ = calibrated_reference_outputs(
                wrapper, result, change
            )
        else:
            point_scores, point_targets, point_probabilities = (
                result.scores,
                result.targets,
                result.probabilities,
            )
        scores.append(tuple(float(v) for v in point_scores[row]))
        if probabilities is not None:
            assert point_probabilities is not None
            probabilities.append(tuple(float(v) for v in point_probabilities[row]))
        targets.append(
            float(point_targets[row])
            if result.task_type == "regression"
            else int(point_targets[row])
        )
    reference = PopulationGraphReference(
        encode(source),
        record_fingerprint(source),
        base.seed,
        base.task_id,
        row,
        feature,
        values,
        anchor,
        schema[feature].vocabulary,
        original.classes,
        capture_location,
        tuple(scores),
        tuple(probabilities) if probabilities is not None else None,
        tuple(targets),
        prior.mode if prior else "original",
        prior.population if prior and prior.mode == "population" else None,
    )

    if wrapper is not None:
        fields: dict[str, Any] = asdict(reference)
        fields["version"] = "calibrated-population-graph-reference-v1"
        return CalibratedPopulationReference(
            **fields,
            raw_scores=tuple(raw_scores),
            calibration_fingerprint=record_fingerprint(asdict(wrapper.calibration)),
        )
    return reference


def evaluate_population_graph_reference(
    source: PopulationReferenceSource
    | CalibratedPopulationGraph
    | CalibratedPopulationQuery
    | CalibratedPopulationIntervention,
    *,
    row: int,
    feature: int,
    values: tuple[float, ...],
    capture_location: str = "capture",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationGraphReference:
    """Recompute a retained row's descendants; never draw a new row-owned forest.

    All observed values are anchored simultaneously. Numeric channels cut state
    and readout. Native labels cut their emitted value and descendant encoding.
    Typed converter capture-only versus representation cuts remain explicit.
    Every evaluation uses the original full row shape and keyed streams, then
    selects this original row. Probabilities are the actual target adapter/domain
    probabilities; targets retain the same stochastic label/noise draw. Neither
    field is a GP posterior, conditional mean or newly sampled population.
    """
    from r0.modules.priors.lib.calibrated_population_reference import CALIBRATED_REFERENCE_TYPES

    if type(source) not in (
        RecursivePopulationGraph,
        RecursivePopulationGraphIntervention,
        *CALIBRATED_REFERENCE_TYPES,
    ):
        raise ValueError("population reference requires an explicit retained graph population")
    if type(limits) is not CausalLimits:
        raise ValueError("population reference requires explicit causal limits")
    function_record_cost(source, limits)
    return population_reference_from_source(
        asdict(source),
        row=row,
        feature=feature,
        values=values,
        capture_location=capture_location,
        limits=limits,
    )


def _decode_reference(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PopulationGraphReference:
    """Verify complete reference identity/derived values with no query fitting."""
    admit_function_metadata(data, limits)
    from r0.modules.priors.lib.calibrated_population_reference import CalibratedPopulationReference

    cls = (
        CalibratedPopulationReference
        if type(data) is dict
        and cast(dict[str, Any], data).get("version") == "calibrated-population-graph-reference-v1"
        else PopulationGraphReference
    )
    raw = complete_population_record(data, cls)
    from r0.modules.priors.lib.population_reference_metadata import validate_reference_metadata

    source = validate_reference_metadata(raw, limits)
    result = population_reference_from_source(
        source,
        row=raw["row"],
        feature=raw["feature"],
        values=tuple(raw["values"]),
        capture_location=raw["capture_location"],
        limits=limits,
    )
    if encode(asdict(result)) != encode(raw):
        raise ValueError("population reference differs from retained source and exact query replay")
    return result


def population_reference_from_source(
    source: Any,
    *,
    row: int,
    feature: int,
    values: tuple[float, ...],
    capture_location: str = "capture",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationGraphReference:
    """Validate complete query/source metadata before any retained construction replay."""
    try:
        return _reference_from_source(
            source,
            row=row,
            feature=feature,
            values=values,
            capture_location=capture_location,
            limits=limits,
        )
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population reference source or controls") from exc


def population_graph_reference_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PopulationGraphReference:
    """Reopen a reference with exact source identity, query semantics and derived arrays."""
    try:
        return _decode_reference(data, limits=limits)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population reference receipt") from exc
