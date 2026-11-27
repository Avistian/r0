"""Whole-row typed reference and comparison admission before RNG or custody replay."""

from dataclasses import asdict, fields, replace

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    SoftIntervention,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.intervention_policy_laws import numeric_leaves, parameter_group
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.typed_policy_locations import PolicyClamp
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def admit_typed_policy(request: TypedInterventionPolicyRequest) -> None:
    from r0.modules.priors.lib.typed_policy_decoding import raw_reference_preflight

    function_record_cost(
        request,
        replace(
            request.limits,
            working_bytes=min(request.limits.working_bytes, request.raw_limits.working_bytes),
        ),
    )
    if 6 * request.raw_limits.state_bytes > min(
        request.raw_limits.working_bytes, request.limits.working_bytes
    ):
        raise ValueError("typed policy initial state reservation exceeds declared limits")
    raw_reference_preflight(asdict(request.base), request.reference_rows, request.limits)
    graph = request.graph
    effective = CausalLimits(
        **{
            f.name: min(getattr(graph.limits, f.name), getattr(request.limits, f.name))
            for f in fields(CausalLimits)
        }
    )
    bounded = replace(graph, limits=effective)
    rows = request.reference_rows if request.kind == "hard" else graph.task.rows
    if isinstance(bounded, FunctionalCausalRecipe):
        reference = bounded.admit(evaluation_rows=rows)
    else:
        reference = replace(
            bounded,
            task=replace(
                bounded.task,
                rows=max(rows, 2),
                context_rows=min(bounded.task.context_rows, max(rows, 2) - 1),
            ),
        ).admit()
    if request.kind == "hard":
        changes: list[PolicyClamp] = []
        for address in request.locations:
            vocabulary = address.vocabulary(graph.nodes[address.node])
            changes.append(
                address.clamp(
                    graph.nodes[address.node],
                    () if vocabulary else (0.0,) * (len(address.channels) or 1),
                    vocabulary[0] if vocabulary else None,
                )
            )
        comparison = CausalInterventionRecipe(
            bounded, tuple(changes), request.mode, request.population
        ).admit()
    else:
        comparison = CausalInterventionRecipe(
            bounded,
            tuple(
                SoftIntervention(i, function=request.equations[i].function)
                for i in request.eligible
            ),
            request.mode,
            request.population,
        ).admit()
    _, storage = function_record_cost(request, request.limits)
    widths = sum(n.function.input_width + n.function.output_width for n in request.equations)
    roots = sum(
        function_record_cost(n.root, request.limits)[0]
        for n in request.equations
        if n.root is not None
    )
    numbers = 64 * rows * (widths + roots + len(request.locations)) if request.kind == "hard" else 0
    numbers += sum(
        256 * max(rows, 256) * n.function.output_width
        for n in request.equations
        if request.kind == "hard" and n.exogenous.inputs.bound is not None
    )
    numbers += sum(
        numeric_leaves(parameter_group(request.equations[x.node].function, x.path)[2])
        for x in request.parameter_laws
    )
    numbers += len(request.eligible) + 1 + sum(len(x.channels) or 1 for x in request.locations)
    draws = (
        64 * (len(graph.nodes) + widths + len(request.locations)) + len(request.parameter_laws) + 2
    )
    draws += sum(
        256 * n.function.output_width
        for n in request.equations
        if request.kind == "hard" and n.exogenous.inputs.bound is not None
    )
    working = (
        storage
        + reference["estimated_working_bytes"]
        + comparison["estimated_working_bytes"]
        + 128 * numbers
        + 2048 * draws
        + 6 * request.raw_limits.state_bytes
    )
    if (
        numbers > request.raw_limits.numbers
        or draws > request.raw_limits.draws
        or working > min(request.limits.working_bytes, request.raw_limits.working_bytes)
    ):
        raise ValueError("typed policy aggregate reference/comparison/raw admission exceeds limits")
    RawDrawLedger(limits=request.raw_limits).admit_specs(
        (RawDrawSpec(("typed-policy",), "aggregate-reserve", "normal", (numbers,)),)
    )
