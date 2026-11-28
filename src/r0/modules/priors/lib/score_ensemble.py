"""Shared-input score ensembles compiled before the ordinary task adapter."""

import math
from dataclasses import replace

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.network_recipe import NetworkRecipe

_DEFAULT = CausalLimits()


def score_ensemble_recipe(
    members: tuple[FunctionalCausalRecipe | NetworkRecipe, ...],
    *,
    weights: tuple[float, ...] = (),
    reduction: str = "mean",
    limits: CausalLimits = _DEFAULT,
) -> FunctionalCausalRecipe:
    """Combine scalar/vector scores, then apply one shared target/noise adapter.

    Members must declare the same task, root and root-only observed channels.
    Structural noise and additional roots are rejected: merging their streams
    would otherwise silently change member semantics. Frozen noisy fits remain
    ordinary deterministic functions. Signed weights support additive scores.
    """
    if (
        type(members) is not tuple
        or not 1 <= len(members) <= 128
        or any(type(m) not in (FunctionalCausalRecipe, NetworkRecipe) for m in members)
        or type(weights) is not tuple
        or reduction not in ("mean", "sum")
        or type(limits) is not CausalLimits
    ):
        raise ValueError("invalid bounded score ensemble controls")
    if not weights:
        weights = (1.0,) * len(members)
    if len(weights) != len(members) or any(not finite_number(w) for w in weights):
        raise ValueError("ensemble requires one finite weight per member")
    try:
        denominator = math.fsum(weights) if reduction == "mean" else 1.0
    except OverflowError as exc:
        raise ValueError("ensemble weight sum overflows") from exc
    if denominator == 0 or not math.isfinite(denominator):
        raise ValueError("mean ensemble requires a finite nonzero weight sum")
    coefficients = tuple(w / denominator for w in weights)
    if any(not math.isfinite(w) for w in coefficients):
        raise ValueError("normalized ensemble weights overflow")
    recipes = tuple(m.runtime if isinstance(m, NetworkRecipe) else m for m in members)
    first = recipes[0]
    for member in recipes:
        if (
            member.version != "functional-causal-v1"
            or member.task != first.task
            or member.nodes[0] != first.nodes[0]
            or member.observed != first.observed
            or any(i != 0 for i, _ in member.observed)
            or any(not n.parents for n in member.nodes[1:])
            or any(any(n.noise_scale) for n in member.nodes)
        ):
            raise ValueError(
                "ensemble members require identical task/root and noise-free shared inputs"
            )
    outputs = len(first.target)
    count = 2 + sum(len(m.nodes) - 1 for m in recipes)
    channels = (
        first.nodes[0].function.output_width
        + outputs
        + sum(n.function.output_width for m in recipes for n in m.nodes[1:])
    )
    if count > limits.nodes or channels > limits.total_channels:
        raise ValueError("ensemble aggregate geometry exceeds limits")
    nodes = [first.nodes[0]]
    targets: list[tuple[tuple[int, int], ...]] = []
    for member in recipes:
        offset = len(nodes) - 1
        mapping = (0,) + tuple(i + offset for i in range(1, len(member.nodes)))
        nodes.extend(
            replace(n, parents=tuple(mapping[p] for p in n.parents)) for n in member.nodes[1:]
        )
        targets.append(tuple((mapping[i], j) for i, j in member.target))
    parents = tuple(dict.fromkeys(i for refs in targets for i, _ in refs))
    offsets: dict[int, int] = {}
    width = 0
    for parent in parents:
        offsets[parent] = width
        width += nodes[parent].function.output_width
    if width * outputs > limits.parameters or 65536 + 512 * width * outputs > limits.working_bytes:
        raise ValueError("ensemble reduction matrix exceeds limits")
    matrix = [[0.0] * width for _ in range(outputs)]
    for coefficient, refs in zip(coefficients, targets, strict=True):
        for output, (node, channel) in enumerate(refs):
            matrix[output][offsets[node] + channel] += coefficient
    index = len(nodes)
    nodes.append(
        FunctionalCausalNode(parents, FrozenLinearFunction(tuple(tuple(row) for row in matrix)))
    )
    return FunctionalCausalRecipe(
        first.task, tuple(nodes), first.observed, tuple((index, j) for j in range(outputs)), limits
    )
