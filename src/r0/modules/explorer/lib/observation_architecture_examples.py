"""Explicit r0 observation extensions retain each broad architecture's own source laws."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors as p


def architecture_observation_examples(task: p.LinearRecipe) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    source_task = replace(task, task_type="multiclass", classes=4)
    for family in (
        "dense",
        "individual-sparse",
        "block-sparse",
        "skip",
        "residual",
        "irregular",
        "gated",
    ):
        prior = asdict(p.NetworkSamplingRequest(2, (5, 3, 4), families=(family,)))
        prior.pop("limits")
        examples[f"Observations: r0 general / variable-width {family}"] = {
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
                "profile": "r0-propagated-block-v1",
                "selection": "unrestricted",
                "target": "internal",
                "feature_order": "permuted",
                "raw_limits": asdict(p.RawDrawLimits()),
            },
        }
    item = deepcopy(examples["Observations: r0 general / variable-width gated"])
    first = p.NetworkBlock(
        (0,), (p.NetworkAffine(((1.0, 0.0), (0.0, 1.0), (1.0, -1.0)), (0.1, -0.1, 0.2)),)
    )
    branch = p.NetworkAffine(
        ((1.0, 0.0, 0.5, 2.0, -1.0), (0.0, 1.0, -1.0, 0.5, 2.0)),
        (0.2, -0.3),
        mask=((True, False, True, True, True), (False, True, True, True, True)),
        activation=p.CausalActivation("tanh"),
    )
    second = p.NetworkBlock(
        (1, 0),
        (branch, replace(branch, multiplier=-0.5)),
        gate=p.NetworkAffine(((0.1, -0.2, 0.3, 0.1, 0.0), (-0.1, 0.2, 0.0, 0.0, 0.2)), (0.0, 0.1)),
        skips=(p.NetworkProjection(0, p.NetworkAffine(((0.5, 0.0), (0.0, -0.5)), (0.0, 0.0))),),
    )
    final = p.NetworkBlock((2, 1), (p.NetworkAffine(((1.0, -2.0, 0.5, 1.0, -0.5),), (0.3,)),))
    item["source"] = {
        "topology": "network-authored",
        "task": asdict(task),
        "root": None,
        "network": asdict(p.FrozenNetworkFunction(2, (first, second, final))),
    }
    examples["Observations: r0 general / authored sparse gated multi-parent skips"] = item
    item = deepcopy(item)
    item["source"] = {
        "topology": "network-innovations",
        "source": item["source"],
        "statistics": asdict(
            p.NetworkInnovationRequest(
                p.NetworkScalarPrior("uniform", 0.05, 0.3),
                importance=p.NetworkScalarPrior("uniform", 0.5, 1.5),
                placement="post-block",
            )
        ),
        "plan": None,
    }
    examples["Observations: r0 general / combined architecture + post-block noise"] = item
    item = deepcopy(examples["Observations: r0 general / variable-width skip"])
    item["source"] = {
        "topology": "source-root-network",
        "source": item["source"],
        "profile": asdict(p.SourceRootProfile(2, selector="p1-code-mixed")),
        "modes": ["identity", "identity"],
        "calibration_rows": 0,
        "root_seed": "7",
        "raw_limits": asdict(p.RawDrawLimits()),
    }
    examples["Observations: r0 general / skip + P1 mixed root"] = item
    item = deepcopy(examples["Observations: r0 general / variable-width irregular"])
    item["source"] = {
        "topology": "prototype-root-network",
        "source": item["source"],
        "lifecycle": "frozen-independent-prototype-bank-r0-v1",
        "reference_rows": 32,
        "rho": 0.125,
        "count_rule": "ceil-r0",
        "max_prototypes": 256,
        "weights": asdict(p.PrototypeWeightLaw()),
        "raw_limits": asdict(p.RawDrawLimits()),
        "bank_seed": "13",
    }
    examples["Observations: r0 general / irregular + independent prototype root"] = item
    item = deepcopy(examples["Observations: r0 general / variable-width gated"])
    hyper = p.NetworkHyperpriorRequest(
        p.NetworkSamplingRequest(2, (4,), families=("gated",)),
        depth=p.NetworkScalarPrior("fixed", 2.0, 2.0),
        width=p.NetworkScalarPrior("integer-uniform", 3.0, 6.0),
        weight_scale=p.NetworkScalarPrior("uniform", 0.1, 0.5),
        width_scope="per-hidden-block",
        max_hidden_blocks=2,
        max_hidden_width=6,
    )
    raw = asdict(hyper)
    raw["template"].pop("limits")
    item["source"] = {
        "topology": "network-hyperprior",
        "task": asdict(source_task),
        "root": None,
        "hyperprior": raw,
    }
    examples["Observations: r0 general / gated + independently sampled widths"] = item
    return examples
