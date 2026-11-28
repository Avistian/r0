"""Ordered encoded parent states and observed readouts for terminal population forests."""

from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converted_evaluation import evaluate_converted
from r0.modules.priors.lib.converted_output import converted_value
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng, functional_values
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import evaluate_native_graph
from r0.modules.priors.lib.native_vector_evaluation import evaluate_native_vector_graph
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.training_features import TrainingFeature

type PopulationAncestors = (
    FunctionalCausalRecipe | NativeCausalRecipe | NativeVectorRecipe | ConvertedCausalRecipe
)
ANCESTOR_TYPES = (
    FunctionalCausalRecipe,
    NativeCausalRecipe,
    NativeVectorRecipe,
    ConvertedCausalRecipe,
)


@dataclass(frozen=True, slots=True)
class PopulationInputs:
    states: tuple[torch.Tensor, ...]
    observed: torch.Tensor
    schema: tuple[TrainingFeature, ...]


def population_ancestor_values(
    ancestors: PopulationAncestors,
    *,
    seed: int,
    task_id: int,
    intervention: CausalInterventionRecipe | None = None,
) -> PopulationInputs:
    """Do not evaluate an unused ancestor target adapter or calibrate on population rows."""
    execution = intervention_execution(intervention) if intervention else None
    runtime = intervention.effective_runtime() if intervention else ancestors
    if isinstance(runtime, ConvertedCausalRecipe):
        assert execution is None or isinstance(execution, ConverterInterventionExecution)
        result = evaluate_converted(runtime, seed=seed, task_id=task_id, execution=execution)
        return PopulationInputs(
            result.states,
            torch.stack(
                [converted_value(result, r).to(torch.float64) for r in runtime.observed], 1
            ),
            tuple(runtime.nodes[r.node].feature(r) for r in runtime.observed),
        )
    if isinstance(runtime, NativeVectorRecipe):
        vector = evaluate_native_vector_graph(
            runtime, seed=seed, task_id=task_id, execution=execution
        )
        return PopulationInputs(
            vector.states,
            torch.stack([vector.values[i][j].to(torch.float64) for i, j in runtime.observed], 1),
            tuple(
                TrainingFeature(
                    runtime.nodes[i].domains[j].kind, runtime.nodes[i].domains[j].vocabulary
                )
                for i, j in runtime.observed
            ),
        )
    if isinstance(runtime, NativeCausalRecipe):
        native = evaluate_native_graph(runtime, seed=seed, task_id=task_id, execution=execution)
        return PopulationInputs(
            native.states,
            torch.stack([native.values[i][:, j].to(torch.float64) for i, j in runtime.observed], 1),
            tuple(
                TrainingFeature(runtime.nodes[i].domain.kind, runtime.nodes[i].domain.vocabulary)
                for i, _ in runtime.observed
            ),
        )
    readouts: list[torch.Tensor] = []

    def rng(*keys: object) -> np.random.Generator:
        return (
            execution.rng(lambda *k: functional_rng(seed, task_id, *k), *keys)
            if execution
            else functional_rng(seed, task_id, *keys)
        )

    states = functional_values(runtime, rng, "cpu", readouts=readouts, execution=execution)
    return PopulationInputs(
        tuple(states),
        torch.stack([readouts[i][:, j] for i, j in runtime.observed], 1),
        (TrainingFeature(),) * len(runtime.observed),
    )


def population_state_widths(ancestors: PopulationAncestors) -> tuple[int, ...]:
    if isinstance(ancestors, NativeVectorRecipe):
        return tuple(n.state_width for n in ancestors.nodes)
    if isinstance(ancestors, NativeCausalRecipe):
        return tuple(n.state_width for n in ancestors.nodes)
    if isinstance(ancestors, ConvertedCausalRecipe):
        return tuple(n.equation.function.output_width for n in ancestors.nodes)
    return tuple(n.function.output_width for n in ancestors.nodes)
