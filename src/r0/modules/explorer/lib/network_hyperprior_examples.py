"""Separate paper/code scalar-law presets on explicit r0 architecture templates."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors


def network_hyperprior_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    task = priors.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    for source, label in (("p1-paper", "P1 paper"), ("tabicl-9530bd6", "I1 pinned code")):
        for family in (
            "dense",
            "individual-sparse",
            "block-sparse",
            "skip",
            "residual",
            "irregular",
            "gated",
        ):
            template = priors.NetworkSamplingRequest(2, (1,), families=(family,))
            profile = priors.network_source_hyperprior(template, source=source)
            for scope, scope_label in (
                ("shared", "shared width"),
                ("per-hidden-block", "individual widths"),
            ):
                raw = asdict(replace(profile, width_scope=scope))
                raw["template"].pop("limits")
                examples[
                    f"Neural hyperpriors: {label} / {family} / {scope_label} (r0 architecture)"
                ] = {
                    "topology": "network-hyperprior",
                    "task": asdict(task),
                    "root": None,
                    "hyperprior": raw,
                }
    return examples
