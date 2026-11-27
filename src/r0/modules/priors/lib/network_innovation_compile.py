"""Compile frozen network states into explicit keyed functional equations."""

from dataclasses import replace

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.network_block_function import FrozenNetworkBlockFunction
from r0.modules.priors.lib.network_innovation_plan import NetworkInnovationPlan
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.structural_network import NetworkAffine, NetworkBlock, NetworkProjection


def _importance(branch: NetworkAffine, factors: tuple[float, ...]) -> NetworkAffine:
    return replace(
        branch,
        matrix=tuple(
            tuple(w * f for w, f in zip(row, factors, strict=True)) for row in branch.matrix
        ),
    )


def _identity(
    width: int, bias: tuple[float, ...], activation: CausalActivation | None = None
) -> FrozenNetworkBlockFunction:
    matrix = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    return FrozenNetworkBlockFunction(
        (width,),
        NetworkBlock(
            (0,), (NetworkAffine(matrix, bias, activation=activation or CausalActivation()),)
        ),
    )


def compile_network_innovations(
    source: NetworkRecipe, plan: NetworkInnovationPlan
) -> tuple[tuple[FunctionalCausalNode, ...], tuple[int, ...]]:
    if type(source) is not NetworkRecipe or type(plan) is not NetworkInnovationPlan:
        raise ValueError("network innovations require a frozen standalone source and plan")
    f = source.function
    if (
        len(plan.stages) != len(f.blocks)
        or plan.importance
        and len(plan.importance) != f.input_width
    ):
        raise ValueError("network innovation stages/importance must match source widths")
    # Bound generated identity matrices and retained duplicated source before building any.
    parameters, storage = function_record_cost((source, plan), source.limits)
    extra = sum(
        2 * b.output_width**2 + 8 * b.output_width
        for b, s in zip(f.blocks, plan.stages, strict=True)
        if s is not None
    )
    if (
        3 * parameters + extra > source.limits.parameters
        or 3 * storage + 256 * extra > source.limits.working_bytes
    ):
        raise ValueError("compiled network innovation metadata exceeds declared admission")
    count = (
        1
        + len(f.blocks)
        + sum(1 + int(s.placement == "pre-activation") for s in plan.stages if s is not None)
    )
    if count > source.limits.nodes:
        raise ValueError("compiled network innovation node count exceeds declared admission")
    nodes = [source.runtime.nodes[0]]
    mapping = [0]
    widths = [f.input_width]
    for index, (original, stage) in enumerate(zip(f.blocks, plan.stages, strict=True)):
        if stage is not None and len(stage.scales) != original.output_width:
            raise ValueError("network innovation channel count differs from block width")
        if plan.profile != "r0-explicit" and (
            len(original.experts) != 1 or original.gate is not None or original.skips
        ):
            raise ValueError("source placement profiles require single-expert unskipped blocks")
        if plan.profile != "r0-explicit" and original.parents != (index,):
            raise ValueError("source placement profiles require sequential network layers")
        pinned = plan.profile in ("p1-778f0d0-placement", "tabicl-9530bd6-placement")
        if pinned and (
            original.experts[0].placement != "before"
            or index == 0
            and original.experts[0].activation != CausalActivation()
        ):
            raise ValueError(
                "pinned code requires first affine then activation-before-affine blocks"
            )
        ancestors = tuple(dict.fromkeys((*original.parents, *(s.source for s in original.skips))))
        ids = {s: i for i, s in enumerate(ancestors)}
        factors = tuple(
            v
            for parent in original.parents
            for v in (
                plan.importance if parent == 0 and plan.importance else (1.0,) * widths[parent]
            )
        )
        experts = (
            tuple(_importance(e, factors) for e in original.experts)
            if plan.importance
            else original.experts
        )
        gate = (
            _importance(original.gate, factors)
            if original.gate and plan.importance_gates
            else original.gate
        )
        skips = tuple(
            NetworkProjection(
                ids[s.source],
                _importance(s.affine, plan.importance)
                if s.source == 0 and plan.importance
                else s.affine,
            )
            for s in original.skips
        )
        activation = CausalActivation()
        if stage is not None and stage.placement == "pre-activation":
            if len(experts) != 1 or gate is not None or skips or experts[0].placement != "after":
                raise ValueError(
                    "pre-activation noise requires one after-activation expert without gate/skips"
                )
            activation = experts[0].activation
            experts = (replace(experts[0], activation=CausalActivation()),)
        block = NetworkBlock(tuple(ids[p] for p in original.parents), experts, gate, skips)
        function = FrozenNetworkBlockFunction(tuple(widths[p] for p in ancestors), block)
        nodes.append(FunctionalCausalNode(tuple(mapping[p] for p in ancestors), function))
        if stage is not None:
            nodes.append(
                FunctionalCausalNode(
                    (len(nodes) - 1,),
                    _identity(original.output_width, stage.means),
                    noise_scale=stage.scales,
                    exogenous=stage.exogenous,
                    noise_response=stage.response,
                )
            )
            if stage.placement == "pre-activation":
                nodes.append(
                    FunctionalCausalNode(
                        (len(nodes) - 1,),
                        _identity(
                            original.output_width, (0.0,) * original.output_width, activation
                        ),
                    )
                )
        mapping.append(len(nodes) - 1)
        widths.append(original.output_width)
    return tuple(nodes), tuple(mapping)
