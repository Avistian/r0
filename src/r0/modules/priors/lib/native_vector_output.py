"""Typed targets and feature schema from logical native vector variables."""

import platform
from dataclasses import asdict

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.native_vector_evaluation import (
    evaluate_native_vector_graph,
    native_vector_rng,
)
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.node_mechanisms import NodeMechanisms
from r0.modules.priors.lib.recursive_function_metadata import recursive_mechanism_metadata
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.root_populations import RootPopulations
from r0.modules.priors.lib.training_features import TrainingFeature
from r0.modules.priors.lib.training_task import TrainingTask


def generate_native_vector(
    recipe: NativeVectorRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
    root_populations: RootPopulations | None = None,
) -> TrainingTask:
    result = evaluate_native_vector_graph(
        recipe,
        seed=seed,
        task_id=task_id,
        device=device,
        execution=execution,
        mechanisms=mechanisms,
        root_populations=root_populations,
    )
    config = asdict(recipe)
    source = {
        **config,
        "task": {
            k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")
        },
    }
    source_id = record_fingerprint((source, seed, task_id))
    target_node, target_channel = recipe.target
    domain = recipe.nodes[target_node].domains[target_channel]
    span = recipe.nodes[target_node].spans[target_channel]
    inputs = torch.stack([result.values[i][j].to(torch.float64) for i, j in recipe.observed], dim=1)
    targets = result.values[target_node][target_channel]
    split_rng = (
        execution.rng(
            lambda *keys: native_vector_rng(seed, task_id, *keys), "split", recipe.task.split
        )
        if execution
        else native_vector_rng(seed, task_id, "split", recipe.task.split)
    )
    permutation = torch.tensor(
        split_rng.permutation(recipe.task.rows),
        device=device,
    )
    emissions = {
        f"{i}:{j}": {
            "domain": asdict(domain),
            "span": asdict(node.spans[j]),
            "uniforms": result.uniforms[i, j].detach().cpu().numpy().tolist()
            if (i, j) in result.uniforms
            else None,
        }
        for i, node in enumerate(recipe.nodes)
        for j, domain in enumerate(node.domains)
        if domain.kind != "continuous"
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
                "family": "native_vector",
                **recursive_mechanism_metadata(tuple(n.equation.function for n in recipe.nodes)),
                "version": recipe.version,
                "target_origin": "native-variable-v1",
                "target": recipe.target,
                "native_emissions": emissions,
                "admission": recipe.admit(),
            }
        ),
        inputs,
        targets,
        targets[:, None]
        if domain.kind == "continuous"
        else result.scores[target_node][:, span.score_start : span.score_stop],
        result.probabilities.get((target_node, target_channel)),
        permutation[: recipe.task.context_rows],
        permutation[recipe.task.context_rows :],
        domain.vocabulary,
        tuple(
            TrainingFeature(recipe.nodes[i].domains[j].kind, recipe.nodes[i].domains[j].vocabulary)
            for i, j in recipe.observed
        ),
        "native-variable-v1",
    )
