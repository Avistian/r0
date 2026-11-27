"""Run an admitted original DAG with explicit per-node population mechanisms."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    SoftIntervention,
)
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converted_output import generate_converted
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import generate_native_causal
from r0.modules.priors.lib.native_vector_output import generate_native_vector
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest
from r0.modules.priors.lib.recursive_population_graph_context import RecursivePopulationContext
from r0.modules.priors.lib.recursive_population_graph_plan import (
    RecursivePopulationGraphRequest,
    validate_population_emission,
)
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class PopulationGraphEvaluation:
    task: TrainingTask
    context: RecursivePopulationContext


def evaluate_population_dag(
    request: RecursivePopulationGraphRequest,
    samples: dict[int, SampledRecursiveForest],
    *,
    seed: int,
    task_id: int,
    intervention: CausalInterventionRecipe | None = None,
) -> PopulationGraphEvaluation:
    execution = intervention_execution(intervention) if intervention else None
    runtime = intervention.effective_runtime() if intervention else request.base
    disabled: frozenset[int] = (
        frozenset(
            c.node
            for c in intervention.interventions
            if isinstance(c, SoftIntervention) and c.function is not None
        )
        if intervention
        else frozenset()
    )
    for binding in request.replacements:
        if binding.node not in disabled:
            validate_population_emission(runtime, binding)
    context = RecursivePopulationContext(samples, disabled=disabled)
    if isinstance(runtime, ConvertedCausalRecipe):
        assert execution is None or isinstance(execution, ConverterInterventionExecution)
        task = generate_converted(
            runtime, seed=seed, task_id=task_id, execution=execution, mechanisms=context
        )
    elif isinstance(runtime, NativeVectorRecipe):
        task = generate_native_vector(
            runtime, seed=seed, task_id=task_id, execution=execution, mechanisms=context
        )
    elif isinstance(runtime, NativeCausalRecipe):
        task = generate_native_causal(
            runtime, seed=seed, task_id=task_id, execution=execution, mechanisms=context
        )
    else:
        task = generate_functional_causal(
            runtime, seed=seed, task_id=task_id, execution=execution, mechanisms=context
        )
    return PopulationGraphEvaluation(task, context)
