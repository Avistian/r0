"""Workbench controls adapt to public sampled-graph construction APIs."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from r0.modules import priors
from r0.modules.explorer.lib.composed_examples import composed_examples, fitted_examples
from r0.modules.explorer.lib.converted import converted_editor, converted_examples
from r0.modules.explorer.lib.innovation_examples import innovation_examples
from r0.modules.explorer.lib.native import (
    native_example,
    native_vector_example,
    sampled_native_example,
)
from r0.modules.explorer.lib.network_hyperprior_examples import network_hyperprior_examples
from r0.modules.explorer.lib.network_innovation_examples import network_innovation_examples
from r0.modules.explorer.lib.networks import network_editor, network_examples
from r0.modules.explorer.lib.prototype_root_examples import prototype_root_examples
from r0.modules.explorer.lib.source_root_examples import source_root_examples


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
        "Native mixed graph (sampled domains)": sampled_native_example(),
        "Native heterogeneous vector graph": native_vector_example(),
        "Native vector parent composition": native_vector_example(parent=True),
        **innovation_examples(geometry),
        **composed_examples(common, geometry, sampled_native_example()),
        **fitted_examples(),
        **network_examples(),
        **source_root_examples(),
        **prototype_root_examples(),
        **network_hyperprior_examples(),
        **network_innovation_examples(),
        **converted_examples(),
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
    if isinstance(recipe, (priors.ConvertedCausalRecipe, priors.SampledConvertedGraph)):
        return converted_editor(
            recipe.recipe if isinstance(recipe, priors.SampledConvertedGraph) else recipe,
            recipe.version,
        )
    if isinstance(recipe, (priors.NetworkRecipe, priors.NetworkInputView)):
        return network_editor(recipe)
    runtime = (
        recipe.recipe
        if isinstance(recipe, (priors.SampledNativeGraph, priors.SampledNativeVectorGraph))
        else getattr(recipe, "runtime", recipe)
    )
    if isinstance(runtime, (priors.NativeCausalRecipe, priors.NativeVectorRecipe)):
        return {
            "kind": "mechanism-edit",
            "construction": recipe.version,
            "editable_recipe": asdict(runtime),
            "geometry": None,
            "native": True,
            "vector": isinstance(runtime, priors.NativeVectorRecipe),
            "notice": (
                "Native direct targets. Editing equations or domains creates a linked "
                "authored graph, not a causal intervention. Frequency weights shift logits; "
                "they do not guarantee marginal category counts. Sampled construction "
                "receipts stay in the original archive."
            ),
            "nodes": [
                {
                    "index": i,
                    "parents": list(node.equation.parents),
                    "width": node.value_width,
                    "encoded_width": node.state_width,
                    "score_width": node.equation.function.output_width,
                    "function": node.equation.function.version,
                    "domain": (
                        [asdict(d) for d in node.domains]
                        if isinstance(node, priors.NativeVectorNode)
                        else asdict(node.domain)
                    ),
                }
                for i, node in enumerate(runtime.nodes)
            ],
        }
    fitted = isinstance(
        runtime, (priors.FittedFunctionalCausalRecipe, priors.PopulationFittedGraph)
    )
    if fitted:
        kind = (
            priors.SupportPopulationFunctionalRecipe
            if getattr(runtime, "population", None) == "support-only"
            else priors.FunctionalCausalRecipe
        )
        runtime = kind(
            runtime.task, runtime.nodes, runtime.observed, runtime.target, runtime.limits
        )
    if not isinstance(runtime, priors.FunctionalCausalRecipe):
        raise ValueError("mechanism editing requires a frozen functional causal graph")
    return {
        "kind": "mechanism-edit",
        "construction": recipe.version,
        "editable_recipe": asdict(runtime),
        "geometry": json.loads(task.mechanism_json).get("geometry"),
        "notice": (
            "Editing frozen split tables creates an authored equation; it does not refit. "
            "Synthetic fit populations and backend provenance remain in the original archive. "
            if fitted
            else ""
        )
        + (
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


def frozen_archive_version(version: object) -> bool:
    """Identify archive families whose task identity can advance without resampling."""
    return str(version).startswith("sampled-") or version in (
        "fitted-functional-causal-v1",
        "standalone-network-v1",
        "standalone-network-innovations-v1",
        "population-fitted-functional-causal-v1",
        "network-input-view-v1",
        "network-input-view-v2",
    )


def prepare_mechanism_editor(
    task: priors.TrainingTask, data: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    from r0.modules.explorer.lib.networks import guarded_network_editor

    limits = caller_limits(data)
    prepared = guarded_network_editor(task, data, output_dir, limits=limits)
    return prepared if prepared is not None else mechanism_editor(task, limits=limits)
