"""Population-aware hard-policy references preserve GP roots and fixed-fit lineage."""

from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation, realize_gp_columns
from r0.modules.priors.lib.gp_population_metadata_cost import (
    MetadataCost,
    population_metadata_cost,
    retained_cost,
)
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_intervention import GPRootInterventionSource
from r0.modules.priors.lib.gp_root_intervention_decoding import decode_gp_intervention_source
from r0.modules.priors.lib.gp_root_intervention_metadata import intervention_source_metadata
from r0.modules.priors.lib.gp_root_query_metadata import admit_query_envelope
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.root_populations import RootPopulations
from r0.modules.priors.lib.typed_policy_planning import admit_typed_record
from r0.modules.priors.lib.typed_policy_reference import typed_reference
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest
from r0.modules.priors.lib.typed_policy_request_decoding import typed_request_metadata
from r0.modules.priors.lib.weight_geometry import Matrix


@dataclass(frozen=True, slots=True)
class GPPolicyReference:
    source: GPRootInterventionSource
    policy: TypedInterventionPolicyRequest
    seed: int
    population: GPColumnPopulation
    values: tuple[Matrix | None, ...]
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits
    version: str = "gp-policy-reference-r0-v1"


def realize_gp_policy_reference(
    source: GPRootInterventionSource,
    policy: TypedInterventionPolicyRequest,
    *,
    seed: int,
) -> GPPolicyReference:
    """Evaluate complete reference rows using original or independent same-law GP roots.

    This is a reference for policy bounds, not target calibration or policy sampling.
    Independent references resample root populations, never the retained target fit.
    Fixed coordinate arrays cannot be silently resized; their original row count is
    required. Random/grid coordinate laws can use a different reference row count.
    """
    if type(policy) is not TypedInterventionPolicyRequest or policy.kind != "hard":
        raise ValueError("GP policy reference requires explicit hard-policy controls")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("GP policy reference requires an exact bounded seed")
    limits = policy.limits
    held = retained_cost((source, policy), limits)
    raw_source, raw_policy = asdict(source), asdict(policy)
    metadata = intervention_source_metadata(raw_source, limits)
    if not gp_payload_equal(raw_policy["base"], metadata.current["base"]):
        raise ValueError("GP policy reference must retain its exact authored base graph")
    typed_request_metadata(raw_policy, limits=metadata.limits)
    policy_budget = admit_typed_record(raw_policy, metadata.limits)
    original = metadata.current["source"]
    matched = policy.bounds_lifecycle == "source-reference-population"
    if matched:
        if (seed, policy.reference_seed, policy.reference_task_id, policy.reference_rows) != (
            original["seed"],
            original["seed"],
            original["task_id"],
            metadata.request.rows,
        ):
            raise ValueError("GP source reference requires original identity and complete rows")
        request = metadata.request
    else:
        if (seed, policy.reference_task_id) in metadata.identities:
            raise ValueError("GP independent reference identity collides with source or fit")
        request = replace(metadata.request, rows=policy.reference_rows)
    future = population_metadata_cost(request)
    cost = MetadataCost(
        held.numbers + future.numbers,
        held.storage + future.storage,
        held.nodes + future.nodes,
        max(held.depth, future.depth) + 1,
    )
    population_budget = admit_query_envelope(
        cost, metadata.populations + ((request, metadata.base),), metadata.limits
    )
    if any(
        policy_budget[key] + population_budget[key] > getattr(metadata.limits, limit)
        for key, limit in (
            ("parameters", "parameters"),
            ("estimated_working_bytes", "working_bytes"),
        )
    ):
        raise ValueError("GP policy reference combined population/runtime budget exceeds limits")
    # All identity/geometry/future-population admission precedes any replay or draw.
    valid = decode_gp_intervention_source(raw_source, limits=metadata.limits)
    current = valid if isinstance(valid, GPRootDAGRecipe) else valid.original
    population = (
        current.source
        if matched
        else realize_gp_columns(request, seed=seed, task_id=policy.reference_task_id)
    )
    context = RootPopulations(
        tuple(
            (b.node, tuple(tuple(row[j] for j in b.columns) for row in population.draw.values))
            for b in current.bindings
        ),
        population.population_id,
        population.row_ids,
        tuple((b.node, b.columns) for b in current.bindings),
    )
    arrays, raw = typed_reference(policy, seed, root_populations=context)
    values = tuple(
        None
        if a is None
        else tuple(
            tuple(float(v) for v in row) for row in cast(list[list[float]], cast(Any, a).tolist())
        )
        for a in arrays
    )
    return GPPolicyReference(valid, policy, seed, population, values, raw, metadata.limits)
