"""Original-shape replay under observed cuts, retaining GP and adapter draws."""

from typing import Any

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.converted_output import generate_converted
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import (
    functional_rng,
    generate_functional_causal,
)
from r0.modules.priors.lib.generate import recipe_from_dict
from r0.modules.priors.lib.gp_reference_admission import GPReferencePlan
from r0.modules.priors.lib.gp_reference_runtime import reference_clamps, reference_runtime
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.gp_root_reference import GPRootReference
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import generate_native_causal
from r0.modules.priors.lib.native_vector_output import generate_native_vector
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.targets import adapt_targets


def realize_reference(
    source: dict[str, Any],
    plan: GPReferencePlan,
    *,
    row: int,
    inputs: tuple[tuple[float, ...], ...],
    structural_innovations: str,
    capture_location: str,
) -> GPRootReference:
    retained = recipe_from_dict(source, limits=plan.limits)
    if not isinstance(retained, (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery)):
        raise ValueError("unsupported GP reference source")
    body = retained if isinstance(retained, GPRootDAGRecipe) else retained.original
    runtime = reference_runtime(body.base, structural_innovations)
    seed, task_id = body.source.seed, body.source.task_id
    context = body.context()
    raw_rows: list[tuple[float, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    probability_rows: list[tuple[float, ...]] = []
    targets: list[float | int] = []
    classes: tuple[str, ...] = ()
    for point in inputs:
        execution = intervention_execution(
            CausalInterventionRecipe(runtime, reference_clamps(runtime, point, capture_location))
        )
        if isinstance(runtime, FunctionalCausalRecipe):
            task = generate_functional_causal(
                runtime, seed=seed, task_id=task_id, execution=execution, root_populations=context
            )
        elif isinstance(runtime, NativeCausalRecipe):
            task = generate_native_causal(
                runtime, seed=seed, task_id=task_id, execution=execution, root_populations=context
            )
        elif isinstance(runtime, NativeVectorRecipe):
            task = generate_native_vector(
                runtime, seed=seed, task_id=task_id, execution=execution, root_populations=context
            )
        else:
            assert isinstance(execution, ConverterInterventionExecution)
            task = generate_converted(
                runtime, seed=seed, task_id=task_id, execution=execution, root_populations=context
            )
        scores = task.scores
        if not isinstance(retained, GPRootDAGRecipe):
            scores = retained.calibration.apply(scores)
        if isinstance(runtime, FunctionalCausalRecipe):
            target, probabilities, classes = adapt_targets(
                runtime.task,
                scores,
                noise_rng=functional_rng(seed, task_id, "noise"),
                label_rng=functional_rng(seed, task_id, "labels"),
            )
        else:
            target, probabilities, classes = task.targets, task.probabilities, task.classes
            if not classes:
                target = scores[:, 0]
        raw_rows.append(tuple(float(v) for v in task.scores[row]))
        score_rows.append(tuple(float(v) for v in scores[row]))
        if probabilities is not None:
            probability_rows.append(tuple(float(v) for v in probabilities[row]))
        targets.append(int(target[row]) if classes else float(target[row]))
    return GPRootReference(
        encode(source),
        record_fingerprint(source),
        body.source.population_id,
        plan.fit_population_id,
        seed,
        task_id,
        row,
        inputs,
        plan.schema,
        structural_innovations,
        capture_location,
        tuple(raw_rows),
        tuple(score_rows),
        None if classes else tuple(v[0] for v in score_rows),
        tuple(probability_rows) if probability_rows else None,
        tuple(targets),
        classes,
    )
