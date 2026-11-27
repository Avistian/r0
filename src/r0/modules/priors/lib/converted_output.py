"""Direct typed captured targets, without numeric target adapters."""

import platform
from dataclasses import asdict

import numpy as np
import torch

from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe, ConvertedReadout
from r0.modules.priors.lib.converted_evaluation import (
    ConvertedEvaluation,
    converted_rng,
    evaluate_converted,
)
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def converted_value(result: ConvertedEvaluation, ref: ConvertedReadout) -> torch.Tensor:
    if ref.capture is not None:
        return result.emissions[ref.node, ref.capture].categories
    assert ref.channel is not None
    return result.readouts[ref.node][:, ref.channel]


def generate_converted(
    recipe: ConvertedCausalRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: ConverterInterventionExecution | None = None,
) -> TrainingTask:
    result = evaluate_converted(
        recipe, seed=seed, task_id=task_id, device=device, execution=execution
    )
    config = asdict(recipe)
    source = {
        **config,
        "task": {
            k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")
        },
    }
    source_id = record_fingerprint((source, seed, task_id))
    inputs = torch.stack(
        [converted_value(result, ref).to(torch.float64) for ref in recipe.observed], dim=1
    )
    targets = converted_value(result, recipe.target)
    feature = recipe.nodes[recipe.target.node].feature(recipe.target)
    emission = result.emissions.get((recipe.target.node, recipe.target.capture or ""))
    permutation = torch.tensor(
        converted_rng(seed, task_id, "split", recipe.task.split).permutation(recipe.task.rows),
        device=device,
    )
    emissions = {
        f"{i}:{name}": {
            "uniforms": result.uniforms[i, name].cpu().numpy().tolist()
            if (i, name) in result.uniforms
            else None,
            "categories": value.categories.cpu().numpy().tolist(),
            "probabilities": value.probabilities.cpu().numpy().tolist(),
        }
        for (i, name), value in result.emissions.items()
    }
    return TrainingTask(
        source_id,
        record_fingerprint((source_id, recipe.task.split, recipe.task.context_rows)),
        recipe.task.task_type,
        encode(
            {
                "recipe": config,
                "seed": seed,
                "task_id": task_id,
                "execution": {
                    "implementation": recipe.version,
                    "device": device,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": str(torch.__version__),
                    "cpu_threads": torch.get_num_threads(),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                },
            }
        ),
        encode(
            {
                "family": "converted_causal",
                "version": recipe.version,
                "target_origin": "native-variable-v1",
                "target": asdict(recipe.target),
                "converter_emissions": emissions,
                "admission": recipe.admit(),
                "capture_stage": "before structural innovation and node importance",
                "state_width": "converter input width; no one-hot expansion",
            }
        ),
        inputs,
        targets,
        emission.scores if emission else targets[:, None],
        emission.probabilities if emission else None,
        permutation[: recipe.task.context_rows],
        permutation[recipe.task.context_rows :],
        feature.vocabulary,
        tuple(recipe.nodes[ref.node].feature(ref) for ref in recipe.observed),
        "native-variable-v1",
    )
