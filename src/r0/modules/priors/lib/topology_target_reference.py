"""Enclosing target adapters on a frozen topology's explicitly supplied raw reference."""

from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.hyperprior_graph import HyperpriorGraph
from r0.modules.priors.lib.joint_hyperprior_graph import JointHyperpriorGraph
from r0.modules.priors.lib.targets import adapt_targets


def topology_reference_targets(
    recipe: HyperpriorGraph | JointHyperpriorGraph,
    scores: torch.Tensor,
    *,
    seed: int,
    task_id: int,
    row: int,
) -> dict[str, Any]:
    """Use the original row's target draw, never a query-dependent population refit.

    Scores are raw topology causal-reference readouts. Probabilities are the
    enclosing adapter probabilities conditional on that explicit latent clamp,
    not an observational conditional distribution. Regression means average only
    the independent target noise; realized views hold its original draw fixed.
    """
    if (
        type(recipe) not in (HyperpriorGraph, JointHyperpriorGraph)
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or type(row) is not int
        or not 0 <= row < recipe.task.rows
        or type(scores) is not torch.Tensor
        or scores.ndim != 2
        or scores.dtype != torch.float64
        or scores.device.type != "cpu"
        or not 2 <= scores.shape[0] <= 512
        or scores.shape[1] != (recipe.task.classes if recipe.task.task_type == "multiclass" else 1)
    ):
        raise ValueError("topology target reference requires exact identity and aligned raw scores")
    recipe.runtime.admit(evaluation_rows=recipe.task.rows + len(scores))
    if not torch.isfinite(scores).all():
        raise ValueError("topology reference scores must be finite")
    recipe.validate()
    noise_rng = functional_rng(seed, task_id, "noise")
    label_rng = functional_rng(seed, task_id, "labels")
    noise = float(noise_rng.normal(size=recipe.task.rows)[row])
    uniform = float(label_rng.uniform(size=recipe.task.rows)[row])
    targets, probabilities, classes = adapt_targets(
        recipe.task,
        scores,
        noise_rng=noise_rng,
        label_rng=label_rng,
        fixed_noise=torch.full((len(scores),), noise, dtype=torch.float64),
        fixed_uniforms=torch.full((len(scores),), uniform, dtype=torch.float64),
    )
    regression = recipe.task.task_type == "regression"
    return {
        "target_reference_version": "topology-target-reference-v1",
        "task_type": recipe.task.task_type,
        "classes": classes,
        "target_probabilities": None
        if probabilities is None
        else np.asarray(probabilities.numpy(), dtype=np.float64).tolist(),
        "target_mean": np.asarray(scores[:, 0].numpy(), dtype=np.float64).tolist()
        if regression
        else None,
        "target_standard_deviation": recipe.task.noise_ratio if regression else None,
        "fixed_draw_targets": np.asarray(
            targets.numpy(), dtype=np.float64 if regression else np.int64
        ).tolist(),
        "fixed_target_noise": noise if regression else None,
        "fixed_label_uniform": None if regression else uniform,
        "target_semantics": (
            "enclosing adapter on explicit causal readouts; target-noise mean and original-row "
            "fixed-draw realization, not observational conditioning or a new query population"
        ),
    }
