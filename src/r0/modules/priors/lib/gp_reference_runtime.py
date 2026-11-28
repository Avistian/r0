"""Explicit observed-value clamps shared by all four GP-root runtime families."""

from dataclasses import replace

from r0.modules.priors.lib.causal_intervention_records import HardIntervention
from r0.modules.priors.lib.converted_causal_records import (
    ConvertedCausalNode,
    ConvertedCausalRecipe,
)
from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention
from r0.modules.priors.lib.exogenous import NoiseResponse
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.gp_root_dags import GPRootRuntime
from r0.modules.priors.lib.native_causal import NativeCausalNode, NativeCausalRecipe
from r0.modules.priors.lib.native_vector_intervention import NativeVectorHardIntervention
from r0.modules.priors.lib.native_vector_records import NativeVectorNode, NativeVectorRecipe
from r0.modules.priors.lib.training_features import TrainingFeature

type ReferenceHard = HardIntervention | NativeVectorHardIntervention | ConverterHardIntervention


def observed_schema(runtime: GPRootRuntime) -> tuple[TrainingFeature, ...]:
    if isinstance(runtime, ConvertedCausalRecipe):
        return tuple(runtime.nodes[ref.node].feature(ref) for ref in runtime.observed)
    if isinstance(runtime, NativeVectorRecipe):
        return tuple(
            TrainingFeature(
                runtime.nodes[i].domains[j].kind, runtime.nodes[i].domains[j].vocabulary
            )
            for i, j in runtime.observed
        )
    if isinstance(runtime, NativeCausalRecipe):
        return tuple(
            TrainingFeature(runtime.nodes[i].domain.kind, runtime.nodes[i].domain.vocabulary)
            for i, _ in runtime.observed
        )
    return (TrainingFeature(),) * runtime.task.features


def reference_clamps(
    runtime: GPRootRuntime, anchor: tuple[float, ...], capture_location: str
) -> tuple[ReferenceHard, ...]:
    """All observed values anchored, with explicit capture versus representation semantics."""
    schema = observed_schema(runtime)
    changes: list[ReferenceHard] = []
    numeric: dict[int, list[tuple[int, float]]] = {}
    for column, (value, feature) in enumerate(zip(anchor, schema, strict=True)):
        category = None
        if feature.vocabulary:
            if int(value) != value or not 0 <= value < len(feature.vocabulary):
                raise ValueError("GP reference categories require declared integer codes")
            category = feature.vocabulary[int(value)]
        if isinstance(runtime, ConvertedCausalRecipe):
            ref = runtime.observed[column]
            if ref.capture is not None:
                assert category is not None
                changes.append(
                    ConverterHardIntervention(ref.node, ref.capture, category, capture_location)
                )
            else:
                assert ref.channel is not None
                numeric.setdefault(ref.node, []).append((ref.channel, value))
        elif isinstance(runtime, NativeVectorRecipe):
            node, variable = runtime.observed[column]
            changes.append(
                NativeVectorHardIntervention(
                    node, variable, value=value if category is None else None, category=category
                )
            )
        elif isinstance(runtime, NativeCausalRecipe):
            node, channel = runtime.observed[column]
            if category is None:
                numeric.setdefault(node, []).append((channel, value))
            else:
                changes.append(HardIntervention(node, category=category))
        else:
            node, channel = runtime.observed[column]
            numeric.setdefault(node, []).append((channel, value))
    for node, entries in numeric.items():
        changes.append(
            HardIntervention(
                node,
                tuple(c for c, _ in entries),
                tuple(v for _, v in entries),
                location="state"
                if isinstance(runtime, NativeCausalRecipe)
                else "state_and_readout",
            )
        )
    return tuple(changes)


def reference_runtime(runtime: GPRootRuntime, policy: str) -> GPRootRuntime:
    if policy == "original-row":
        return runtime
    nodes: list[
        FunctionalCausalNode | NativeCausalNode | NativeVectorNode | ConvertedCausalNode
    ] = []
    for node in runtime.nodes:
        equation = node if isinstance(node, FunctionalCausalNode) else node.equation
        quiet = replace(
            equation,
            noise_scale=(0.0,) * equation.function.output_width,
            noise_response=NoiseResponse(),
        )
        nodes.append(
            quiet if isinstance(node, FunctionalCausalNode) else replace(node, equation=quiet)
        )
    return replace(runtime, nodes=tuple(nodes))
