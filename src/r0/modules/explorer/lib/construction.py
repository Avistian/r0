"""Workbench controls adapt to public sampled-graph construction APIs."""

from dataclasses import asdict, replace
from typing import Any, cast

from r0.modules import priors
from r0.modules.explorer.lib.native import native_example


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
    geometry = {
        **{k: v for k, v in common.items() if k not in ("widths", "observed", "target")},
        "subsample": True,
        "width_prior": asdict(priors.JointWidthPrior()),
        "noise_scale": 0.0,
    }
    return {
        "Explicit causal graph": {**common, "topology": "explicit", "parents": [[], [0], [0, 1]]},
        "Cauchy causal graph": {**common, "topology": "cauchy", "offset": 0.0},
        "Geometry-sampled causal graph": {
            **geometry,
            "topology": "geometry",
            "nodes": 3,
            "offset": 0.0,
        },
        "Attachment geometry (r0 extension)": {
            **geometry,
            "topology": "attachment-geometry",
            "topology_prior": asdict(priors.AttachmentGraphPrior(nodes=3)),
        },
        "Motif geometry (r0 adaptation)": {
            **geometry,
            "topology": "motif-geometry",
            "topology_prior": asdict(priors.MotifGraphPrior(depth=1)),
        },
        "Native mixed graph (authored)": native_example(),
        **{
            f"Numerical pipeline: {variant}": {
                **geometry,
                "topology": "geometry",
                "nodes": 3,
                "offset": 0.0,
                "pipeline_prior": asdict(
                    priors.JointPipelinePrior(
                        converter=replace(priors.JointNumericalConverterPrior(), variant=variant)
                    )
                ),
            }
            for variant in ("historical-tuple", "historical-graph", "readout-only")
        },
    }


def mechanism_editor(task: priors.TrainingTask, *, limits: priors.CausalLimits) -> dict[str, Any]:
    import json

    raw = json.loads(task.recipe_json)["recipe"]
    recipe = priors.recipe_from_dict(raw, limits=limits)
    runtime = getattr(recipe, "runtime", recipe)
    if isinstance(runtime, priors.NativeCausalRecipe):
        return {
            "kind": "mechanism-edit",
            "construction": runtime.version,
            "editable_recipe": asdict(runtime),
            "geometry": None,
            "native": True,
            "notice": (
                "Native direct targets. Editing equations or domains creates a linked "
                "authored graph, not a causal intervention."
            ),
            "nodes": [
                {
                    "index": i,
                    "parents": list(node.equation.parents),
                    "width": node.value_width,
                    "encoded_width": node.state_width,
                    "score_width": node.equation.function.output_width,
                    "function": node.equation.function.version,
                    "domain": asdict(node.domain),
                }
                for i, node in enumerate(runtime.nodes)
            ],
        }
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
