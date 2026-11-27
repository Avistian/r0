"""Broad observation policies expose source distinctions and supported compositions."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors
from r0.modules.explorer.lib.observation_architecture_examples import (
    architecture_observation_examples,
)


def observation_examples() -> dict[str, Any]:
    task = priors.LinearRecipe(
        rows=32, context_rows=20, features=2, signal_features=2, active_features=2
    )
    source_task = replace(task, task_type="multiclass", classes=6)
    examples: dict[str, Any] = {}
    for profile, label in (
        ("p1-paper-block-r0-v1", "P1 paper + explicit r0 laws"),
        ("p1-778f0d0-v1", "P1 pinned 778f0d0"),
    ):
        for family in ("dense", "individual-sparse", "block-sparse"):
            prior = asdict(priors.NetworkSamplingRequest(2, (6, 6, 6), families=(family,)))
            prior.pop("limits")
            examples[f"Observations: {label} / {family}"] = {
                "topology": "network-observations",
                "task": asdict(task),
                "source": {
                    "topology": "network",
                    "task": asdict(source_task),
                    "root": None,
                    "network_prior": prior,
                },
                "selection_seed": "17",
                "observations": {
                    "profile": profile,
                    "selection": "block",
                    "target": "final",
                    "feature_order": "preserved",
                    "raw_limits": asdict(priors.RawDrawLimits()),
                },
            }
    base = next(iter(examples.values()))
    for selection, target in (
        ("unrestricted", "internal"),
        ("unrestricted", "final"),
        ("block", "internal"),
    ):
        item = deepcopy(base)
        item["observations"].update(selection=selection, target=target, feature_order="permuted")
        examples[f"Observations: explicit r0 {selection} / {target} / permuted"] = item
    for placement in ("pre-activation", "post-block"):
        item = deepcopy(base)
        request = priors.NetworkInnovationRequest(
            priors.NetworkScalarPrior("uniform", 0.05, 0.3),
            importance=priors.NetworkScalarPrior("uniform", 0.5, 1.5),
        )
        item["source"] = {
            "topology": "network-innovations",
            "source": item["source"],
            "statistics": asdict(replace(request, placement=placement)),
            "plan": None,
        }
        examples[f"Observations: internal noise + importance / {placement} (r0)"] = item
    item = deepcopy(base)
    profile = priors.network_source_hyperprior(
        priors.NetworkSamplingRequest(2, (6,), families=("dense",)), source="p1-paper"
    )
    profile = replace(
        profile,
        depth=priors.NetworkScalarPrior("fixed", 2.0, 2.0),
        width=priors.NetworkScalarPrior("fixed", 6.0, 6.0),
        max_hidden_blocks=2,
        max_hidden_width=6,
    )
    raw = asdict(profile)
    raw["template"].pop("limits")
    item["source"] = {
        "topology": "network-hyperprior",
        "task": asdict(source_task),
        "root": None,
        "hyperprior": raw,
    }
    examples["Observations: P1 scalar weight prior / explicit fixed equal widths"] = item
    for selector in ("p1-code-mixed", "p2-node-mixture-r0"):
        item = deepcopy(base)
        item["source"] = {
            "topology": "source-root-network",
            "source": item["source"],
            "profile": asdict(priors.SourceRootProfile(2, selector=selector)),
            "modes": ["identity", "identity"],
            "calibration_rows": 0,
            "root_seed": "7",
            "raw_limits": asdict(priors.RawDrawLimits()),
        }
        examples[f"Observations: {selector} coordinate root / identity"] = item
    item = deepcopy(base)
    item["source"] = {
        "topology": "prototype-root-network",
        "source": item["source"],
        "lifecycle": "frozen-independent-prototype-bank-r0-v1",
        "reference_rows": 32,
        "rho": 0.125,
        "count_rule": "ceil-r0",
        "max_prototypes": 256,
        "weights": asdict(priors.PrototypeWeightLaw()),
        "raw_limits": asdict(priors.RawDrawLimits()),
        "bank_seed": "13",
    }
    examples["Observations: independent prototype root / r0 count and weight laws"] = item
    item = deepcopy(base)
    first = priors.NetworkBlock(
        (0,),
        (
            priors.NetworkAffine(
                ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0), (0.0, 2.0)),
                (0.0,) * 6,
            ),
        ),
    )
    matrix = tuple(tuple(float(i == j) * 2 for j in range(6)) for i in range(6))
    blocks = (
        first,
        *(
            priors.NetworkBlock((i,), (priors.NetworkAffine(matrix, (float(i),) * 6),))
            for i in (1, 2)
        ),
    )
    item["source"] = {
        "topology": "network-authored",
        "task": asdict(source_task),
        "root": None,
        "network": asdict(priors.FrozenNetworkFunction(2, blocks)),
    }
    item["observations"]["profile"] = "p1-778f0d0-v1"
    examples["Observations: authored affine equations / pinned selection"] = item
    examples.update(architecture_observation_examples(task))
    return examples
