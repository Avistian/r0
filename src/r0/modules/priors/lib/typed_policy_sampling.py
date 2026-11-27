"""Atomic typed selection, categorical choices and frozen reference-bound realization."""

from dataclasses import asdict

import numpy as np

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    SoftIntervention,
)
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions, recorded_draw
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches
from r0.modules.priors.lib.typed_policy_admission import admit_typed_policy
from r0.modules.priors.lib.typed_policy_locations import PolicyClamp
from r0.modules.priors.lib.typed_policy_parameters import resample_typed_function
from r0.modules.priors.lib.typed_policy_records import (
    SampledTypedInterventionPolicy,
    TypedInterventionBounds,
)
from r0.modules.priors.lib.typed_policy_reference import typed_reference
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def sample_typed_intervention_policy(
    request: TypedInterventionPolicyRequest,
    *,
    selection_rng: np.random.Generator,
    value_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
) -> SampledTypedInterventionPolicy:
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.typed_policy_decoding import restore_typed_base

    if type(request) is not TypedInterventionPolicyRequest:
        raise ValueError("typed sampling requires an explicit frozen policy request")
    admit_typed_policy(request)
    raw_base = asdict(request.base)
    recipe_from_dict(raw_base, limits=request.limits)
    replay_base = restore_typed_base(raw_base)
    if not construction_matches(replay_base, raw_base):
        raise ValueError("typed policy base differs from strict original construction custody")
    ledger = RawDrawLedger(limits=request.raw_limits)
    streams = {"selection": selection_rng, "values": value_rng, "calibration": calibration_rng}
    with raw_transactions(streams, ledger) as (staged, capture):
        assert capture is not None
        initial = tuple(
            snapshot_generator(staged[k], byte_limit=request.raw_limits.state_bytes)
            for k in streams
        )
        order = recorded_draw(
            staged["selection"],
            capture,
            owner=("typed-policy",),
            role="eligible-permutation",
            distribution="permutation",
            shape=(len(request.eligible),),
            parameters=(("count", len(request.eligible)),),
        )
        selected = tuple(request.eligible[int(i)] for i in order[: request.selected_count])
        seed = request.reference_seed
        reference_raw = ()
        bounds: list[TypedInterventionBounds] = []
        changes: list[PolicyClamp | SoftIntervention] = []
        if request.kind == "hard":
            numeric = any(not a.vocabulary(request.graph.nodes[a.node]) for a in request.locations)
            arrays = []
            if numeric:
                if request.bounds_lifecycle == "independent-calibration-r0":
                    seed = int(
                        recorded_draw(
                            staged["calibration"],
                            capture,
                            owner=("typed-policy",),
                            role="calibration-seed",
                            distribution="integers",
                            shape=(1,),
                            parameters=(("high", 2**63 - 1),),
                        )[0]
                    )
                arrays, reference_raw = typed_reference(request, seed)
            for i, address in enumerate(request.locations):
                if address.node not in selected:
                    continue
                node = request.graph.nodes[address.node]
                vocabulary = address.vocabulary(node)
                if vocabulary:
                    u = float(
                        recorded_draw(
                            staged["values"],
                            capture,
                            owner=("typed-policy", f"location:{i}"),
                            role="category-quantile-r0",
                            distribution="random",
                            shape=(1,),
                        )[0]
                    )
                    probabilities = np.asarray(
                        address.probabilities or (1.0 / len(vocabulary),) * len(vocabulary)
                    )
                    cumulative = probabilities.cumsum()
                    cumulative[-1] = 1.0
                    category = vocabulary[int(np.searchsorted(cumulative, u, side="right"))]
                    changes.append(address.clamp(node, category=category))
                else:
                    value = arrays[i]
                    assert value is not None
                    lo = value.min(dim=0).values.numpy()
                    hi = value.max(dim=0).values.numpy()
                    u = recorded_draw(
                        staged["values"],
                        capture,
                        owner=("typed-policy", f"location:{i}"),
                        role="bound-quantile",
                        distribution="random",
                        shape=(len(lo),),
                    )
                    sampled = (1 - u) * lo + u * hi
                    if not np.isfinite(sampled).all():
                        raise ValueError("nonfinite typed hard realization; no retries")
                    bounds.append(
                        TypedInterventionBounds(
                            i, tuple(float(x) for x in lo), tuple(float(x) for x in hi)
                        )
                    )
                    changes.append(address.clamp(node, tuple(float(x) for x in sampled)))
        else:
            changes.extend(
                SoftIntervention(i, function=resample_typed_function(request, i, staged, capture))
                for i in selected
            )
        runtime = CausalInterventionRecipe(
            request.base, tuple(changes), request.mode, request.population
        )
        runtime.admit()
        final = tuple(
            snapshot_generator(staged[k], byte_limit=request.raw_limits.state_bytes)
            for k in streams
        )
        result = SampledTypedInterventionPolicy(
            request,
            selected,
            tuple(bounds),
            seed,
            runtime,
            initial,
            final,
            capture.freeze(),
            reference_raw,
        )
    return result
