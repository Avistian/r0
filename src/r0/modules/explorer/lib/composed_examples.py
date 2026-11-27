"""Inspectable parent-composition and independently fitted-tree starting points."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from r0.modules import priors


def composed_examples(
    common: dict[str, Any], geometry: dict[str, Any], native: dict[str, Any]
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for source in ("tabiclv2", "mitra2", "extension"):
        examples[f"Parent composition: {source} explicit"] = {
            **common,
            "topology": "explicit",
            "parents": [[], [0], [0, 1]],
            "composition": asdict(priors.ParentCompositionPrior(source=source)),
        }
    base = {
        **geometry,
        "topology": "geometry",
        "nodes": 3,
        "offset": 0.0,
        "composition": asdict(priors.ParentCompositionPrior()),
    }
    examples["Parent composition: geometry"] = base
    for variant in ("historical-tuple", "historical-graph", "readout-only"):
        examples[f"Parent composition: pipeline {variant}"] = {
            **base,
            "pipeline_prior": asdict(
                priors.JointPipelinePrior(
                    converter=priors.JointNumericalConverterPrior(variant=variant),
                )
            ),
        }
    examples["Parent composition: native domains"] = {
        **native,
        "version": "native-parent-graph-request-v1",
        "composition": asdict(priors.ParentCompositionPrior(source="mitra2", aggregation="mean")),
    }
    return deepcopy(examples)


def fitted_examples() -> dict[str, Any]:
    skeleton = priors.FunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=64, context_rows=48, features=2, signal_features=2, active_features=2
        ),
        (
            priors.FunctionalCausalNode(
                (),
                priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
                root=priors.JointRootBase("gaussian", 2),
            ),
            priors.FunctionalCausalNode((0,), priors.FrozenLinearFunction(((1.0, -1.0),))),
            priors.FunctionalCausalNode((0, 1), priors.FrozenLinearFunction(((1.0, 1.0, 1.0),))),
        ),
        ((0, 0), (0, 1)),
        ((2, 0),),
    )
    result: dict[str, Any] = {}
    for family in ("dt", "extra_trees", "rf", "gbdt"):
        result[f"Fitted structural tree: {family}"] = {
            "topology": "fitted",
            "skeleton": asdict(skeleton),
            "fits": [
                {
                    "node": index,
                    "plan": asdict(
                        priors.FittedTreePlan(
                            family=family,
                            input_width=width,
                            rows=64,
                        )
                    ),
                }
                for index, width in ((1, 2), (2, 3))
            ],
        }
    from r0.modules.explorer.lib.fitted_parameter_examples import fitted_parameter_examples
    from r0.modules.explorer.lib.fitted_population_examples import fitted_population_examples

    return {**result, **fitted_parameter_examples(result), **fitted_population_examples(result)}
