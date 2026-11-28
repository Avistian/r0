"""Atomic selection, bounded reference populations and declared parameter draws."""

import json
import math
from dataclasses import asdict, fields, replace

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
    SoftIntervention,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_node_evaluation import evaluate_functional_nodes
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.intervention_policy_function import finish_policy_function
from r0.modules.priors.lib.intervention_policy_laws import (
    numeric_leaves,
    parameter_group,
    replace_leaves,
)
from r0.modules.priors.lib.intervention_policy_records import (
    InterventionBounds,
    SampledInterventionPolicy,
)
from r0.modules.priors.lib.intervention_policy_request import InterventionPolicyRequest
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions, recorded_draw
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec


def admit_intervention_policy(request: InterventionPolicyRequest) -> None:
    if type(request) is not InterventionPolicyRequest:
        raise ValueError("sampling requires a frozen intervention policy request")
    effective = CausalLimits(
        **{
            field.name: min(
                getattr(request.base.limits, field.name), getattr(request.limits, field.name)
            )
            for field in fields(CausalLimits)
        }
    )
    bounded = replace(request.base, limits=effective)
    report = bounded.admit(
        evaluation_rows=request.reference_rows if request.kind == "hard" else request.base.task.rows
    )
    # Admit every eligible change before sampling the subset. Numerical parameter
    # replacement preserves family and geometry; zero clamps have the same shape
    # as sampled constants. This conservative probe never replaces retained records.
    possible: tuple[HardIntervention | SoftIntervention, ...] = tuple(
        HardIntervention(
            index,
            tuple(range(request.base.nodes[index].function.output_width)),
            (0.0,) * request.base.nodes[index].function.output_width,
            location=request.location,
        )
        if request.kind == "hard"
        else SoftIntervention(index, function=request.base.nodes[index].function)
        for index in request.eligible
    )
    comparison = CausalInterventionRecipe(
        bounded, possible, mode=request.mode, population=request.population
    ).admit()
    _parameters, storage = function_record_cost(request, request.limits)
    widths = sum(n.function.input_width + n.function.output_width for n in request.base.nodes)
    # Reserve complete reference raw records plus parameter paths for every eligible
    # choice, not merely the eventually sampled branch. This is a declared conservative
    # bound, not an actual raw draw or a truncated population.
    root_parameters = sum(
        function_record_cost(n.root, request.limits)[0]
        for n in request.base.nodes
        if n.root is not None
    )
    raw_numbers = (
        64 * request.reference_rows * (widths + root_parameters) if request.kind == "hard" else 0
    )
    raw_numbers += sum(
        256 * max(request.reference_rows, 256) * n.function.output_width
        for n in request.base.nodes
        if request.kind == "hard" and n.exogenous.inputs.bound is not None
    )
    raw_numbers += sum(
        numeric_leaves(parameter_group(request.base.nodes[x.node].function, x.path)[2])
        for x in request.parameter_laws
    )
    raw_numbers += (
        len(request.eligible)
        + 1
        + sum(request.base.nodes[i].function.output_width for i in request.eligible)
    )
    draws = (
        64 * (len(request.base.nodes) + widths)
        + len(request.parameter_laws)
        + len(request.eligible)
        + 2
    )
    draws += sum(
        256 * n.function.output_width
        for n in request.base.nodes
        if request.kind == "hard" and n.exogenous.inputs.bound is not None
    )
    working = (
        storage
        + report["estimated_working_bytes"]
        + comparison["estimated_working_bytes"]
        + 128 * raw_numbers
        + 2048 * draws
        + 6 * request.raw_limits.state_bytes
    )
    if raw_numbers > request.raw_limits.numbers or draws > request.raw_limits.draws:
        raise ValueError("policy conservative aggregate raw admission exceeds declared limits")
    if working > min(request.limits.working_bytes, request.raw_limits.working_bytes):
        raise ValueError("policy aggregate working admission exceeds declared limits")
    RawDrawLedger(limits=request.raw_limits).admit_specs(
        (RawDrawSpec(("policy",), "aggregate-reserve", "normal", (raw_numbers,)),)
    )


def sample_intervention_policy(
    request: InterventionPolicyRequest,
    *,
    selection_rng: np.random.Generator,
    value_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
) -> SampledInterventionPolicy:
    admit_intervention_policy(request)
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
            owner=("policy",),
            role="eligible-permutation",
            distribution="permutation",
            shape=(len(request.eligible),),
            parameters=(("count", len(request.eligible)),),
        )
        selected = tuple(request.eligible[int(i)] for i in order[: request.selected_count])
        reference_seed = request.reference_seed
        reference_raw = ()
        bounds: list[InterventionBounds] = []
        changes: list[HardIntervention | SoftIntervention] = []
        if request.kind == "hard":
            if request.bounds_lifecycle == "independent-calibration-r0":
                reference_seed = int(
                    recorded_draw(
                        staged["calibration"],
                        capture,
                        owner=("policy",),
                        role="calibration-seed",
                        distribution="integers",
                        shape=(1,),
                        parameters=(("high", 2**63 - 1),),
                    )[0]
                )
            keys: list[tuple[object, ...]] = []
            for index, node in enumerate(request.base.nodes):
                if node.root is not None:
                    keys.append(("root-base", index))
                for column in range(node.function.output_width):
                    keys.extend(
                        (("innovation", index, column), ("innovation", index, column, "signs"))
                    )
                if node.exogenous.geometry == "ball":
                    keys.append(("innovation", index, "radius"))
            key_names = {key: f"reference:{i}" for i, key in enumerate(keys)}
            prefix = (
                ("policy-independent-reference-v1",)
                if request.bounds_lifecycle == "independent-calibration-r0"
                else ()
            )
            reference_streams = {
                key_names[key]: functional_rng(
                    reference_seed, request.reference_task_id, *prefix, *key
                )
                for key in keys
            }
            reference_ledger = RawDrawLedger(limits=request.raw_limits)
            with raw_transactions(reference_streams, reference_ledger) as (
                references,
                reference_capture,
            ):

                def rng(*key: object):
                    return references[key_names[key]]

                readouts: list[torch.Tensor] = []
                states = evaluate_functional_nodes(
                    request.base.nodes,
                    request.reference_rows,
                    rng,
                    "cpu",
                    readouts=readouts,
                    capture=reference_capture,
                    owner=("policy", "reference"),
                    limits=request.limits,
                )
            reference_raw = reference_ledger.freeze()
            for node in selected:
                table = readouts[node] if request.location == "readout" else states[node]
                lo, hi = table.amin(dim=0).numpy(), table.amax(dim=0).numpy()
                bounds.append(
                    InterventionBounds(
                        node, tuple(float(x) for x in lo), tuple(float(x) for x in hi)
                    )
                )
                u = recorded_draw(
                    staged["values"],
                    capture,
                    owner=("policy", f"node:{node}"),
                    role="hard-quantiles",
                    distribution="random",
                    shape=(len(lo),),
                )
                values = (1 - u) * lo + u * hi
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite hard value; no retry or clipping")
                changes.append(
                    HardIntervention(
                        node,
                        tuple(range(len(lo))),
                        tuple(float(x) for x in values),
                        location=request.location,
                    )
                )
        else:
            for node in selected:
                original = request.base.nodes[node].function
                record = json.loads(json.dumps(asdict(original), allow_nan=False))
                for index, law in enumerate(request.parameter_laws):
                    if law.node != node:
                        continue
                    parent, key, group = parameter_group(record, law.path)
                    size = numeric_leaves(group)
                    raw = recorded_draw(
                        staged["values"],
                        capture,
                        owner=("policy", f"node:{node}"),
                        role=f"parameter:{index}",
                        distribution="normal" if law.family == "normal-r0" else "random",
                        shape=(size,),
                    )
                    if law.family == "normal-r0" or law.family == "uniform-r0":
                        values = law.location + law.scale * raw
                    elif law.family == "log-uniform-r0":
                        values = np.exp(
                            (1 - raw) * math.log(law.location) + raw * math.log(law.scale)
                        )
                    else:
                        values = np.asarray(law.choices)[
                            np.floor(raw * len(law.choices)).astype(np.int64)
                        ]
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite resampled parameter; no retry")
                    parent[key] = replace_leaves(group, iter(values))
                function = finish_policy_function(
                    original,
                    record,
                    tuple(law.path for law in request.parameter_laws if law.node == node),
                    request.limits,
                )
                changes.append(SoftIntervention(node, function=function))
        runtime = CausalInterventionRecipe(
            request.base, tuple(changes), mode=request.mode, population=request.population
        )
        runtime.admit()
        final = tuple(
            snapshot_generator(staged[k], byte_limit=request.raw_limits.state_bytes)
            for k in streams
        )
        result = SampledInterventionPolicy(
            request,
            selected,
            tuple(bounds),
            reference_seed,
            runtime,
            initial,
            final,
            capture.freeze(),
            reference_raw,
        )
    return result
