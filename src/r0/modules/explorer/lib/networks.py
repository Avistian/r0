"""Broad standalone architecture presets and an authored frozen network editor."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors

NOTICE = (
    "Standalone random network: input features feed a frozen architecture, not a causal "
    "graph or parent reducer. Structural masks stay fixed across rows. Zero or disconnected "
    "draws remain valid. Editing creates a linked authored network; original sampling "
    "decisions and raw draws remain in the original archive. It is not an intervention."
)


def network_examples() -> dict[str, Any]:
    task = priors.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    examples: dict[str, Any] = {}
    for family in (
        "dense",
        "individual-sparse",
        "block-sparse",
        "skip",
        "residual",
        "irregular",
        "gated",
    ):
        request = priors.NetworkSamplingRequest(2, (6, 4, 1), families=(family,))
        raw = asdict(request)
        raw.pop("limits")
        examples[f"Standalone network: {family}"] = {
            "topology": "network",
            "task": asdict(task),
            "network_prior": raw,
            "root": None,
        }
    for profile, family in (("p1-individual", "individual-sparse"), ("p1-block", "block-sparse")):
        request = priors.NetworkSamplingRequest(
            2,
            (6, 4, 1),
            families=(family,),
            fan_in=False,
            mask_profile=profile,
            dropout_profile="p1-beta" if profile == "p1-individual" else "fixed",
        )
        raw = asdict(request)
        raw.pop("limits")
        examples[f"Standalone network: {profile} source laws"] = {
            "topology": "network",
            "task": asdict(task),
            "network_prior": raw,
            "root": None,
        }
    left = priors.NetworkAffine(
        ((1.0, -0.5), (-0.3, 0.8)),
        (0.1, -0.2),
        mask=((True, False), (True, True)),
        activation=priors.CausalActivation("tanh"),
    )
    right = priors.NetworkAffine(
        ((-0.7, 0.2), (0.4, -1.0)), (0.0, 0.3), activation=priors.CausalActivation("relu")
    )
    gate = priors.NetworkAffine(((0.8, 0.0), (0.0, -0.9)), (0.0, 0.0))
    skip = priors.NetworkProjection(0, priors.NetworkAffine(((0.2, 0.0), (0.0, 0.2)), (0.0, 0.0)))
    output = priors.NetworkAffine(((0.6, -0.8, 1.2, 0.4),), (0.15,))
    network = priors.FrozenNetworkFunction(
        2,
        (
            priors.NetworkBlock((0,), (left, right), gate, (skip,)),
            priors.NetworkBlock((0, 1), (output,)),
        ),
    )
    examples["Standalone network: sparse + gated + skip (authored)"] = {
        "topology": "network-authored",
        "task": asdict(task),
        "network": asdict(network),
        "root": None,
    }
    return examples


def network_editor(recipe: priors.NetworkRecipe) -> dict[str, Any]:
    editable = replace(recipe, network=recipe.function)
    return {
        "kind": "mechanism-edit",
        "network": True,
        "construction": recipe.version,
        "editable_recipe": asdict(editable),
        "notice": NOTICE,
        "nodes": [
            {
                "index": i,
                "parents": list(b.parents),
                "width": b.output_width,
                "experts": len(b.experts),
                "gated": b.gate is not None,
                "skip_sources": [s.source for s in b.skips],
            }
            for i, b in enumerate(recipe.function.blocks)
        ],
    }


def network_details(recipe: dict[str, Any]) -> dict[str, Any]:
    network = recipe["network"]
    sampled = network.get("version") == "sampled-structural-network-v1"
    function = network["function"] if sampled else network
    return {
        "kind": "standalone-network",
        "notice": NOTICE,
        "sampled": sampled,
        "request": network.get("request"),
        "choices": network.get("choices", []),
        "raw_draw_count": len(network.get("raw", [])),
        "stream_roles": ["named", "raw"] if sampled else [],
        "blocks": len(function["blocks"]),
        "input_width": function["input_width"],
        "output_width": len(function["blocks"][-1]["experts"][0]["bias"]),
        "source_semantics": (
            "Named P1 mask/dropout profiles are source-law adaptations. Residual, skip, irregular "
            "and gated architecture combinations are r0 extensions; no trained labels."
        ),
    }
