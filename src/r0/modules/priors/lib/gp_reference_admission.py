"""Full source, prospective receipt and original-shape execution before GP replay."""

import math
from dataclasses import dataclass, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.gp_population_metadata_cost import (
    MetadataCost,
    record_cost,
    retained_cost,
    sequence_cost,
)
from r0.modules.priors.lib.gp_reference_runtime import (
    observed_schema,
    reference_clamps,
    reference_runtime,
)
from r0.modules.priors.lib.gp_root_dags import GPRootRuntime
from r0.modules.priors.lib.gp_root_decoding import preflight_gp_root_dag
from r0.modules.priors.lib.gp_root_query_metadata import admit_query_envelope, query_source_metadata
from r0.modules.priors.lib.gp_root_reference import GPRootReference
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.training_features import TrainingFeature


@dataclass(frozen=True)
class GPReferencePlan:
    limits: CausalLimits
    runtime: GPRootRuntime
    schema: tuple[TrainingFeature, ...]
    score_width: int
    classes: tuple[str, ...]
    current: dict[str, Any]
    fit_population_id: str | None


def target_geometry(base: GPRootRuntime) -> tuple[int, tuple[str, ...]]:
    if isinstance(base, FunctionalCausalRecipe):
        return len(base.target), (
            ()
            if base.task.task_type == "regression"
            else tuple(f"class-{i}" for i in range(base.task.classes))
        )
    if isinstance(base, ConvertedCausalRecipe):
        feature = base.nodes[base.target.node].feature(base.target)
        return max(1, len(feature.vocabulary)), feature.vocabulary
    node, channel = base.target
    if isinstance(base, NativeVectorRecipe):
        selected = base.nodes[node]
        domain, span = selected.domains[channel], selected.spans[channel]
        return span.score_stop - span.score_start, domain.vocabulary
    assert isinstance(base, NativeCausalRecipe)
    selected = base.nodes[node]
    return (
        1 if selected.domain.kind == "continuous" else selected.equation.function.output_width
    ), selected.domain.vocabulary


def admit_gp_reference(
    source: Any,
    *,
    row: int,
    inputs: tuple[tuple[float, ...], ...],
    structural_innovations: str,
    capture_location: str,
    limits: CausalLimits,
) -> GPReferencePlan:
    if type(limits) is not CausalLimits:
        raise ValueError("GP reference requires explicit caller limits")
    source_cost = retained_cost(source, limits)
    if type(source) is not dict:
        raise ValueError("GP reference requires a complete source object")
    source = cast(dict[str, Any], source)
    if source.get("version") == "gp-root-dag-v1":
        cap, base, _, request = preflight_gp_root_dag(source, limits=limits)
        current, populations, fit_id = source, ((request, base),), None
    elif source.get("version") in ("gp-root-calibration-v1", "gp-root-calibration-query-v1"):
        metadata = query_source_metadata(source, limits=limits)
        cap, base, request, current, populations = (
            metadata.limits,
            metadata.base,
            metadata.request,
            metadata.current,
            metadata.populations,
        )
        ancestor = source
        while ancestor["version"] == "gp-root-calibration-query-v1":
            ancestor = ancestor["source"]
        fit_id = ancestor["fit"]["source"]["population_id"]
    else:
        raise ValueError("unsupported retained GP reference source")
    if (
        type(row) is not int
        or not 0 <= row < base.task.rows
        or type(inputs) is not tuple
        or not 1 <= len(inputs) <= 129
        or base.task.rows * len(inputs) > 20000
        or structural_innovations not in ("original-row", "zero")
        or capture_location not in ("capture", "representation_and_capture")
    ):
        raise ValueError("invalid GP reference row, grid or explicit policies")
    schema = observed_schema(base)
    for point in inputs:
        if type(point) is not tuple or len(point) != len(schema):
            raise ValueError("GP reference inputs must match every observed channel")
        for value, feature in zip(point, schema, strict=True):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("GP reference values require finite float64 coordinates")
            feature.decode(value)
    runtime = reference_runtime(base, structural_innovations)
    width, classes = target_geometry(runtime)
    # Validate every simultaneous typed cut against the metadata-only runtime.
    for point in inputs:
        CausalInterventionRecipe(
            runtime, reference_clamps(runtime, point, capture_location)
        ).admit()
    number = MetadataCost(1, 64, 1, 0)

    def vector(n: int) -> MetadataCost:
        return sequence_cost(number, n)

    def matrix(n: int) -> MetadataCost:
        return sequence_cost(vector(n), len(inputs))

    def text(s: Any) -> MetadataCost:
        return retained_cost(s, cap)

    values = {
        f.name: text(f.default)
        for f in fields(GPRootReference)
        if f.name in ("version", "semantics", "target_draws")
    }
    values.update(
        source_json=MetadataCost(0, 8 * source_cost.storage, 1, 0),
        source_fingerprint=text("0" * 64),
        population_id=text("0" * 64),
        fit_population_id=text(fit_id),
        seed=number,
        task_id=number,
        row=number,
        inputs=matrix(len(schema)),
        feature_schema=retained_cost(schema, cap),
        structural_innovations=text(structural_innovations),
        capture_location=text(capture_location),
        raw_scores=matrix(width),
        scores=matrix(width),
        target_mean=text(None) if classes else vector(len(inputs)),
        probabilities=matrix(len(classes)) if classes else text(None),
        targets=vector(len(inputs)),
        classes=retained_cost(classes, cap),
    )
    future = record_cost(GPRootReference, values)
    # Keep both parsed source and emitted source JSON plus the full original-shape
    # point runtime live. Sequential points retain only bounded output vectors.
    cost = MetadataCost(
        source_cost.numbers + future.numbers,
        source_cost.storage + future.storage,
        source_cost.nodes + future.nodes,
        max(source_cost.depth, future.depth),
    )
    admit_query_envelope(cost, (*populations, (request, runtime)), cap)
    return GPReferencePlan(cap, runtime, schema, width, classes, current, fit_id)
