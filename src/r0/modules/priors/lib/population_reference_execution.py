"""Reference clamps reuse native/typed execution and original full-population row ownership."""

from dataclasses import replace

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
    SoftIntervention,
)
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converted_output import generate_converted
from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import generate_native_causal
from r0.modules.priors.lib.native_vector_intervention import NativeVectorHardIntervention
from r0.modules.priors.lib.native_vector_output import generate_native_vector
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.recursive_population_ancestors import PopulationAncestors
from r0.modules.priors.lib.recursive_population_graph_context import RecursivePopulationContext
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.training_features import TrainingFeature

type ReferenceHard = HardIntervention | NativeVectorHardIntervention | ConverterHardIntervention
type PopulationReferenceSource = RecursivePopulationGraph | RecursivePopulationGraphIntervention


def observed_schema(runtime: PopulationAncestors) -> tuple[TrainingFeature, ...]:
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
    runtime: PopulationAncestors, anchor: tuple[float, ...], capture_location: str
) -> tuple[ReferenceHard, ...]:
    """All observed values anchored, with explicit capture versus representation semantics."""
    schema = observed_schema(runtime)
    changes: list[ReferenceHard] = []
    numeric: dict[int, list[tuple[int, float]]] = {}
    for column, (value, feature) in enumerate(zip(anchor, schema, strict=True)):
        category = None
        if feature.vocabulary:
            if int(value) != value or not 0 <= value < len(feature.vocabulary):
                raise ValueError("population reference categories require declared integer codes")
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


def overlay_reference(
    runtime: PopulationAncestors,
    prior: CausalInterventionRecipe | None,
    clamps: tuple[ReferenceHard, ...],
) -> CausalInterventionRecipe:
    """Reference values override only their explicitly named original hard-cut locations."""
    retained: list[ReferenceHard] = []
    for old in prior.interventions if prior else ():
        if isinstance(old, HardIntervention):
            same = [c for c in clamps if isinstance(c, HardIntervention) and c.node == old.node]
            if old.category is not None:
                if not same:
                    retained.append(old)
                elif any(c.category is None for c in same):
                    raise ValueError("ambiguous numeric/category reference location overlap")
            else:
                if any(c.category is not None for c in same):
                    raise ValueError("ambiguous category/numeric reference location overlap")
                covered = {j for c in same for j in c.channels}
                entries = [
                    (j, v)
                    for j, v in zip(old.channels, old.values, strict=True)
                    if j not in covered
                ]
                if entries:
                    retained.append(
                        replace(
                            old,
                            channels=tuple(j for j, _ in entries),
                            values=tuple(v for _, v in entries),
                        )
                    )
        elif isinstance(old, NativeVectorHardIntervention):
            if not any(
                isinstance(c, NativeVectorHardIntervention)
                and (c.node, c.variable) == (old.node, old.variable)
                for c in clamps
            ):
                retained.append(old)
        elif isinstance(old, ConverterHardIntervention):
            same = [
                c
                for c in clamps
                if isinstance(c, ConverterHardIntervention)
                and (c.node, c.capture) == (old.node, old.capture)
            ]
            if not same:
                retained.append(old)
            elif old.location == "representation_and_capture" and same[0].location == "capture":
                # Two different categories cannot share one retained representation cut.
                if old.category != same[0].category:
                    raise ValueError("capture-only reference overlaps retained representation cut")
                retained.append(old)
                clamps = tuple(c for c in clamps if c is not same[0])
    return CausalInterventionRecipe(
        runtime,
        (*retained, *clamps),
        mode=prior.mode if prior else "matched",
        population=prior.population if prior else 1,
    )


def evaluate_reference_point(
    source: PopulationReferenceSource,
    runtime: PopulationAncestors,
    change: CausalInterventionRecipe,
):
    prior = source.change if isinstance(source, RecursivePopulationGraphIntervention) else None
    base = source.base if isinstance(source, RecursivePopulationGraphIntervention) else source
    samples = (
        source.samples
        if isinstance(source, RecursivePopulationGraphIntervention)
        else tuple(b.sample for b in base.bindings)
    )
    disabled: frozenset[int] = (
        frozenset(
            c.node
            for c in prior.interventions
            if isinstance(c, SoftIntervention) and c.function is not None
        )
        if prior
        else frozenset()
    )
    mechanisms = RecursivePopulationContext(
        {b.node: s for b, s in zip(base.bindings, samples, strict=True)}, disabled=disabled
    )
    execution = intervention_execution(change)
    if isinstance(runtime, ConvertedCausalRecipe):
        assert isinstance(execution, ConverterInterventionExecution)
        return generate_converted(
            runtime,
            seed=base.seed,
            task_id=base.task_id,
            execution=execution,
            mechanisms=mechanisms,
        )
    if isinstance(runtime, NativeVectorRecipe):
        return generate_native_vector(
            runtime,
            seed=base.seed,
            task_id=base.task_id,
            execution=execution,
            mechanisms=mechanisms,
        )
    if isinstance(runtime, NativeCausalRecipe):
        return generate_native_causal(
            runtime,
            seed=base.seed,
            task_id=base.task_id,
            execution=execution,
            mechanisms=mechanisms,
        )
    assert isinstance(runtime, FunctionalCausalRecipe)
    return generate_functional_causal(
        runtime, seed=base.seed, task_id=base.task_id, execution=execution, mechanisms=mechanisms
    )
