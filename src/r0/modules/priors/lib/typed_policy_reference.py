"""Keyed typed reference evaluation retains every root, innovation and emission draw."""

from collections.abc import Callable

import numpy as np
import torch

from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converted_evaluation import converted_rng, evaluate_converted_nodes
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_node_evaluation import evaluate_functional_nodes
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import evaluate_native_nodes, native_rng
from r0.modules.priors.lib.native_vector_evaluation import (
    evaluate_native_vector_nodes,
    native_vector_rng,
)
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def typed_reference(request: TypedInterventionPolicyRequest, seed: int):
    graph = request.graph
    keys: list[tuple[object, ...]] = []
    native = isinstance(graph, (NativeCausalRecipe, NativeVectorRecipe))
    for i, equation in enumerate(request.equations):
        if equation.root is not None:
            keys.append(("root" if native else "root-base", i))
        for c in range(equation.function.output_width):
            keys.extend((("innovation", i, c), ("innovation", i, c, "signs")))
        if equation.exogenous.geometry == "ball":
            keys.append(("innovation", i, "radius"))
        if isinstance(graph, NativeCausalRecipe):
            keys.append(("native-emission", i))
        elif isinstance(graph, NativeVectorRecipe):
            keys.extend(("native-emission", i, v) for v in range(len(graph.nodes[i].domains)))
        elif isinstance(graph, ConvertedCausalRecipe):
            keys.extend(("converter-emission", i, c.name) for c in graph.nodes[i].converters)
    make_rng: Callable[..., np.random.Generator] = functional_rng
    if isinstance(graph, NativeCausalRecipe):
        make_rng = native_rng
    elif isinstance(graph, NativeVectorRecipe):
        make_rng = native_vector_rng
    elif isinstance(graph, ConvertedCausalRecipe):
        make_rng = converted_rng
    prefix = (
        ("typed-policy-independent-reference-v1",)
        if request.bounds_lifecycle == "independent-calibration-r0"
        else ()
    )
    names = {k: str(i) for i, k in enumerate(keys)}
    streams = {names[k]: make_rng(seed, request.reference_task_id, *prefix, *k) for k in keys}
    ledger = RawDrawLedger(limits=request.raw_limits)
    with raw_transactions(streams, ledger) as (staged, capture):

        def rng(*key: object):
            return staged[names[key]]

        arrays: list[torch.Tensor | None] = []
        if isinstance(graph, NativeCausalRecipe):
            scalar_result = evaluate_native_nodes(
                graph.nodes,
                request.reference_rows,
                rng,
                "cpu",
                capture=capture,
                limits=graph.limits,
            )
            arrays = [
                None if a.vocabulary(graph.nodes[a.node]) else scalar_result.values[a.node]
                for a in request.locations
            ]
        elif isinstance(graph, NativeVectorRecipe):
            vector_result = evaluate_native_vector_nodes(
                graph.nodes,
                request.reference_rows,
                rng,
                "cpu",
                capture=capture,
                limits=graph.limits,
            )
            for a in request.locations:
                assert a.variable is not None
                arrays.append(
                    None
                    if a.vocabulary(graph.nodes[a.node])
                    else vector_result.values[a.node][a.variable][:, None]
                )
        else:
            if isinstance(graph, ConvertedCausalRecipe):
                converted_result = evaluate_converted_nodes(
                    graph.nodes,
                    request.reference_rows,
                    rng,
                    "cpu",
                    capture=capture,
                    limits=graph.limits,
                )
                states, readouts = converted_result.states, converted_result.readouts
            else:
                readout_list: list[torch.Tensor] = []
                states = evaluate_functional_nodes(
                    graph.nodes,
                    request.reference_rows,
                    rng,
                    "cpu",
                    readouts=readout_list,
                    capture=capture,
                    limits=graph.limits,
                )
                readouts = tuple(readout_list)
            for a in request.locations:
                arrays.append(
                    None
                    if a.vocabulary(graph.nodes[a.node])
                    else (readouts if a.location == "readout" else states)[a.node][:, a.channels]
                )
    return arrays, ledger.freeze()
