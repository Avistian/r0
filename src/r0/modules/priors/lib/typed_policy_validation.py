"""Shared full outer policy validation for runtime records and metadata-only previews."""

import math
from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    SoftIntervention,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.intervention_policy_laws import (
    InterventionParameterLaw,
    numeric_leaves,
    parameter_group,
)
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.typed_policy_locations import InterventionPolicyLocation, PolicyClamp


def validate_typed_policy_values(
    options: dict[str, Any],
    graph: FunctionalCausalRecipe | NativeCausalRecipe | NativeVectorRecipe | ConvertedCausalRecipe,
    *,
    function_records: tuple[Any, ...] | None = None,
) -> None:
    equations = (
        graph.nodes
        if isinstance(graph, FunctionalCausalRecipe)
        else tuple(n.equation for n in graph.nodes)
    )
    if options["base"].version in (
        "support-population-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
    ):
        raise ValueError(
            "population/support refitting policies remain separate from frozen typed policies"
        )
    if isinstance(graph, ConvertedCausalRecipe):
        targets = {graph.target.node}
    elif isinstance(graph, (NativeCausalRecipe, NativeVectorRecipe)):
        targets = {graph.target[0]}
    else:
        targets = {i for i, _ in graph.target}
    if (
        options["version"] != "typed-intervention-policy-request-v1"
        or options["profile"] != "explicit-typed-policy-r0-v1"
        or options["kind"] not in ("hard", "soft")
        or options["mode"] not in ("matched", "population")
        or options["bounds_lifecycle"]
        not in ("independent-calibration-r0", "source-reference-population")
        or type(options["limits"]) is not CausalLimits
        or type(options["raw_limits"]) is not RawDrawLimits
        or type(options["fraction"]) not in (int, float)
        or not math.isfinite(options["fraction"])
        or not 0 < options["fraction"] <= 1
        or options["count"] is not None
        and (type(options["count"]) is not int or options["count"] < 1)
        or any(
            type(v) is not int or not 0 <= v < 2**63
            for v in (options["reference_seed"], options["reference_task_id"])
        )
        or type(options["population"]) is not int
        or not 1 <= options["population"] < 2**63
        or type(options["reference_rows"]) is not int
        or not 1 <= options["reference_rows"] <= 20000
    ):
        raise ValueError("invalid explicit typed intervention policy")
    for raw_values in (options["label_nodes"], options["eligibility"]):
        values = cast(tuple[Any, ...], raw_values)
        if (
            type(values) is not tuple
            or len(set(values)) != len(values)
            or any(type(i) is not int or not 0 <= i < len(graph.nodes) for i in values)
        ):
            raise ValueError("typed policy node indices must be unique exact integers")
    eligible = options["eligibility"] or tuple(
        i for i in range(len(graph.nodes)) if i not in options["label_nodes"]
    )
    selected_count = (
        options["count"]
        if options["count"] is not None
        else math.ceil(options["fraction"] * len(eligible))
    )
    if (
        set(options["label_nodes"]) != targets
        or options["kind"] == "soft"
        and bool(set(eligible) & targets)
        or not eligible
        or selected_count > len(eligible)
    ):
        raise ValueError(
            "typed policy cannot replace label equations or silently repair eligibility/count"
        )
    if (
        options["bounds_lifecycle"] == "source-reference-population"
        and options["reference_rows"] != graph.task.rows
    ):
        raise ValueError("source reference requires the complete original population")
    locations = cast(tuple[InterventionPolicyLocation, ...], options["locations"])
    laws = cast(tuple[InterventionParameterLaw, ...], options["parameter_laws"])
    if type(locations) is not tuple or any(
        type(x) is not InterventionPolicyLocation for x in locations
    ):
        raise ValueError("typed policy locations must be frozen explicit records")
    if type(laws) is not tuple or any(type(x) is not InterventionParameterLaw for x in laws):
        raise ValueError("typed parameter laws must be frozen explicit records")
    if options["kind"] == "hard":
        if options["parameter_laws"] or {x.node for x in locations} != set(eligible):
            raise ValueError(
                "every eligible hard node requires explicit locations and no soft laws"
            )
        changes: list[PolicyClamp] = []
        for address in locations:
            if address.node in targets:
                if isinstance(graph, ConvertedCausalRecipe):
                    overlaps = (
                        address.kind == "capture"
                        and address.capture == graph.target.capture
                        or address.kind == "numerical"
                        and graph.target.channel in address.channels
                    )
                elif isinstance(graph, (NativeCausalRecipe, NativeVectorRecipe)):
                    overlaps = address.variable == graph.target[1]
                else:
                    overlaps = any(
                        node == address.node and channel in address.channels
                        for node, channel in graph.target
                    )
                if overlaps:
                    raise ValueError(
                        "typed policy cannot directly clamp a declared target variable"
                    )
            vocabulary = address.vocabulary(graph.nodes[address.node])
            values = () if vocabulary else (0.0,) * (len(address.channels) or 1)
            changes.append(
                address.clamp(
                    graph.nodes[address.node], values, vocabulary[0] if vocabulary else None
                )
            )
        CausalInterventionRecipe(
            options["base"], tuple(changes), options["mode"], options["population"]
        )
    else:
        if options["locations"] or {x.node for x in laws} != set(eligible):
            raise ValueError(
                "every eligible soft equation requires explicit laws and no hard locations"
            )
        seen: list[tuple[int, tuple[str, ...]]] = []
        for law in laws:
            if any(
                n == law.node
                and (path[: len(law.path)] == law.path or law.path[: len(path)] == path)
                for n, path in seen
            ):
                raise ValueError("overlapping typed parameter paths")
            try:
                numeric_leaves(
                    parameter_group(
                        function_records[law.node]
                        if function_records is not None
                        else equations[law.node].function,
                        law.path,
                    )[2]
                )
            except (KeyError, IndexError, AttributeError, TypeError) as error:
                raise ValueError("invalid typed parameter path") from error
            seen.append((law.node, law.path))
        CausalInterventionRecipe(
            options["base"],
            tuple(SoftIntervention(i, function=equations[i].function) for i in eligible),
            options["mode"],
            options["population"],
        )
