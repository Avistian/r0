"""Workbench controls adapt to public sampled-graph construction APIs."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules import priors


def caller_limits(data: dict[str, Any]) -> priors.CausalLimits:
    raw = data.get("causal_limits", {})
    if type(raw) is not dict:
        raise ValueError("exploration limits must be an explicit object")
    return priors.CausalLimits(**cast(dict[str, Any], raw))


def construction_examples() -> dict[str, Any]:
    common = {
        "task": asdict(
            priors.LinearRecipe(
                rows=64, context_rows=48, features=2, signal_features=2, active_features=2
            )
        ),
        "widths": [2, 2, 1],
        "observed": [[0, 0], [0, 1]],
        "target": [[2, 0]],
        "calibration_rows": 16,
        "root_families": ["gaussian"],
        "function_prior": asdict(
            priors.JointFunctionPrior(
                functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
            )
        ),
        "mode": "meta",
        "categorical_policy": "shared",
        "global_policy": "scalar-quantile-v1",
    }
    return {
        "Explicit causal graph": {**common, "topology": "explicit", "parents": [[], [0], [0, 1]]},
        "Cauchy causal graph": {**common, "topology": "cauchy", "offset": 0.0},
        "Geometry-sampled causal graph": {
            **{k: v for k, v in common.items() if k not in ("widths", "observed", "target")},
            "topology": "geometry",
            "nodes": 3,
            "offset": 0.0,
            "subsample": True,
            "width_prior": asdict(priors.JointWidthPrior()),
            "noise_scale": 0.0,
        },
    }


def mechanism_editor(task: priors.TrainingTask, *, limits: priors.CausalLimits) -> dict[str, Any]:
    import json

    raw = json.loads(task.recipe_json)["recipe"]
    recipe = priors.recipe_from_dict(raw, limits=limits)
    runtime = getattr(recipe, "runtime", recipe)
    if not isinstance(runtime, priors.FunctionalCausalRecipe):
        raise ValueError("mechanism editing requires a frozen functional causal graph")
    return {
        "kind": "mechanism-edit",
        "construction": recipe.version,
        "editable_recipe": asdict(runtime),
        "geometry": json.loads(task.mechanism_json).get("geometry"),
        "notice": (
            "Editing creates a new authored mechanism linked to the original sampled archive. "
            "It is not a causal intervention or observational conditioning."
        ),
        "nodes": [
            {
                "index": i,
                "parents": list(node.parents),
                "width": node.function.output_width,
                "function": node.function.version,
                "observed": [j for n, j in runtime.observed if n == i],
                "target": [j for n, j in runtime.target if n == i],
            }
            for i, node in enumerate(runtime.nodes)
        ],
    }
