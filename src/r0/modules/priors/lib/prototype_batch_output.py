"""Batch-reference emission separates population construction from fixed replay."""

from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.network_sampling_records import SampledNetwork
from r0.modules.priors.lib.prototype_batch_network import (
    PrototypeBatchNetworkRecipe,
    RealizedPrototypeBatchNetworkRecipe,
    realize_prototype_batch_network,
)
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def generate_prototype_batch_network(
    recipe: PrototypeBatchNetworkRecipe | RealizedPrototypeBatchNetworkRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
) -> TrainingTask:
    if any(type(value) is not int or not 0 <= value < 2**63 for value in (seed, task_id)):
        raise ValueError("batch population identities require exact integers in [0,2**63)")
    request = recipe.request if isinstance(recipe, RealizedPrototypeBatchNetworkRecipe) else recipe
    request.admit()
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("batch network requires an available explicit cpu/cuda device")
    if isinstance(request.source.network, SampledNetwork):
        request.source.network.validate_replay()
    realized = (
        recipe
        if isinstance(recipe, RealizedPrototypeBatchNetworkRecipe)
        and (seed, task_id) == (recipe.seed, recipe.task_id)
        else realize_prototype_batch_network(request, seed=seed, task_id=task_id)
    )
    realized.__post_init__()
    inputs = torch.tensor(realized.population.draw.values, dtype=torch.float64, device=device)
    diagnostics: list[dict[str, Any]] = []
    scores = evaluate_admitted_function(request.source.function, inputs, diagnostics=diagnostics)
    card: dict[str, Any] = {
        "version": realized.version,
        "family": "prototype_batch_network",
        "lifecycle": "explicit batch reference; population n determines prototype count",
        "population_identity": {"seed": seed, "task_id": task_id, "rows": request.task.rows},
        "population_digest": record_fingerprint(asdict(realized.population)),
        "prototype_count": realized.population.root.count,
        "normalization": "none at prototype output; construction-base calibration remains explicit",
        "moment_scope": (
            "conditional on the retained batch bank; not unconditional construction variance"
        ),
        "active": list(range(request.task.features)),
        "node_diagnostics": diagnostics,
        "flags": sorted({flag for item in diagnostics for flag in item["flags"]}),
        "resource_admission": request.admit(),
    }
    source_config = asdict(realized)
    source_config["request"]["source"]["task"] = {
        k: v for k, v in asdict(request.task).items() if k not in ("context_rows", "split")
    }
    return complete_task(
        request.task,
        asdict(realized),
        card,
        inputs,
        scores,
        seed=seed,
        task_id=task_id,
        rng=lambda *keys: functional_rng(seed, task_id, *keys),
        source_config=source_config,
    )
