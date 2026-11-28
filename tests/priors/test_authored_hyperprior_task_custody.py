"""Authored topology inspection binds the whole task before selecting rows."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("mutation", ["task_type", "classes", "dtype", "unselected_score", "split"])
def test_authored_hyperprior_rejects_forged_task(mutation: str):
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    edge = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)))
    request = p.HyperpriorGraphRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FixedSizeTopologyHyperprior(5),
        root,
        edge,
    )
    recipe = p.sample_hyperprior_graph(request, rng=np.random.default_rng(43))
    task = p.generate_task(recipe, seed=11, task_id=13)
    assert p.explain_generation(task, [7, 1, 7])["target_max_error"] == 0
    if mutation == "task_type":
        forged = replace(task, task_type="binary")
    elif mutation == "classes":
        forged = replace(task, classes=2)
    elif mutation == "dtype":
        forged = replace(task, inputs=task.inputs.float())
    elif mutation == "unselected_score":
        scores = task.scores.clone()
        scores[15, 0] += 1
        forged = replace(task, scores=scores)
    else:
        forged = replace(task, context_indices=task.context_indices.flip(0))
    with pytest.raises(ValueError, match="hyperprior trace"):
        p.explain_generation(forged, [7, 1, 7])
