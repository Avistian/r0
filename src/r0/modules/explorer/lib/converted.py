"""Typed converter controls keep captured categories separate from descendant state."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors


def converted_examples() -> dict[str, Any]:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    roots = priors.JointRootPrior(families=("gaussian",), function=functions)
    request = priors.ConvertedGraphRequest(
        priors.AttachmentGraphPrior(nodes=3),
        (("red", "green", "blue"), (), ("north", "south", "east", "west", "central")),
        ("no", "yes"),
        rows=64,
        context_rows=48,
        calibration_rows=32,
        width_prior=priors.JointWidthPrior(low=1, high=4),
        root_prior=roots,
        function_prior=functions,
        converter_prior=priors.CategoricalConverterPrior(embeddings=roots),
        pipeline_prior=priors.JointPipelinePrior(
            converter=priors.JointNumericalConverterPrior(
                allow_activation=False, allow_kumaraswamy=False
            )
        ),
    )
    return {
        "Typed categorical converter pipelines": {
            "topology": "sampled-converted",
            "request": asdict(request),
        },
        "Typed converters + innovations (r0 extension)": {
            "topology": "sampled-converted",
            "request": asdict(
                replace(
                    request,
                    pipeline_prior=priors.JointInnovationPipelinePrior(
                        channel_importance=False,
                        l2_normalization=False,
                        converter=priors.JointNumericalConverterPrior(
                            allow_activation=False, allow_kumaraswamy=False
                        ),
                        innovation=priors.JointInnovationPrior(
                            degrees=(3.0,), strengths=(0.1,), powers=(1.0,), truncation=(None,)
                        ),
                    ),
                )
            ),
        },
    }


def converted_editor(recipe: priors.ConvertedCausalRecipe, construction: str) -> dict[str, Any]:
    return {
        "kind": "mechanism-edit",
        "construction": construction,
        "editable_recipe": asdict(recipe),
        "geometry": None,
        "converted": True,
        "notice": (
            "Editing creates a linked authored mechanism. It is not a causal "
            "intervention. Original fit archives remain retained. Categories "
            "are captured before structural noise and node importance; "
            "descendants use the converter representation."
        ),
        "nodes": [
            {
                "index": i,
                "parents": list(n.equation.parents),
                "width": n.equation.function.output_width,
                "function": n.equation.function.version,
                "converters": [asdict(c) for c in n.converters],
            }
            for i, n in enumerate(recipe.nodes)
        ],
    }


def converted_target_view(record: dict[str, Any]) -> dict[str, Any]:
    original = record["recipe"]
    base = original.get("base", original)
    runtime = base["recipe"] if base["version"] == "sampled-converted-graph-v1" else base
    target = runtime["target"]
    capture = target.get("capture")
    converter = next(
        (c for c in runtime["nodes"][target["node"]]["converters"] if c["name"] == capture), None
    )
    original_converter = converter
    forced = None
    for change in original.get("interventions", []):
        if change.get("node") == target["node"] and change.get("capture") == capture:
            if change["version"] == "converter-soft-intervention-v1":
                converter = change["converter"]
            elif change["version"] == "converter-hard-intervention-v1":
                forced = change
    result: dict[str, Any] = {
        "converted": True,
        "target_emission": "numeric pipeline readout",
        "target_frequency_weights": None,
        "target_converter": converter,
        "original_target_converter": original_converter,
        "target_converter_intervention": forced,
        "target_reference": target,
        "capture_stage": (
            "before structural innovation and node importance; descendants use final numeric state"
        ),
    }
    if converter is not None:
        result["target_emission"] = (
            "softmax" if converter["mode"].startswith("softmax_") else "nearest prototype"
        )
    if forced is not None:
        result["target_emission"] = "forced category indicator (not original scores)"
    return result
