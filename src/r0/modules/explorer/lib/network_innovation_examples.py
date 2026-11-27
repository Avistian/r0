"""Visible variety in architecture, statistics, source placement and authored laws."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p
from r0.modules.explorer.lib.network_hyperprior_examples import network_hyperprior_examples
from r0.modules.explorer.lib.networks import network_examples


def network_innovation_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    request = p.NetworkInnovationRequest(
        p.NetworkScalarPrior("log-uniform", 0.01, 0.4),
        mean=p.NetworkScalarPrior("uniform", -0.1, 0.1),
        importance=p.NetworkScalarPrior("choice", choices=(0.0, 0.3, 1.0, 3.0)),
        scale_scope="per-channel",
        mean_scope="per-block",
        channel_scale="half-normal",
    )
    original = network_examples()
    for name, source in original.items():
        examples[name.replace("Standalone network:", "Noisy network:")] = {
            "topology": "network-innovations",
            "source": source,
            "statistics": asdict(request),
            "plan": None,
        }
    # Existing scalar hyperprior variety composes directly, rather than being a
    # disconnected sampler catalog. Default source operational caps stay explicit.
    for name, source in network_hyperprior_examples().items():
        examples["Noisy " + name] = {
            "topology": "network-innovations",
            "source": source,
            "statistics": asdict(request),
            "plan": None,
        }
    task = p.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    for profile in ("p1-paper", "p1-778f0d0", "tabicl-9530bd6"):
        paper = profile == "p1-paper"
        first = p.NetworkAffine(
            ((0.8, -0.3), (0.2, 0.9)),
            (0.1, -0.2),
            activation=p.CausalActivation("tanh") if paper else p.CausalActivation(),
            placement="after" if paper else "before",
        )
        second = p.NetworkAffine(
            ((0.7, -0.5),),
            (0.2,),
            activation=p.CausalActivation("tanh"),
            placement="after" if paper else "before",
        )
        network = p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (first,)), p.NetworkBlock((1,), (second,)))
        )
        stats = p.network_source_innovations(
            source=profile,
            pre_sample_scales=not paper,
            mean=p.NetworkScalarPrior("uniform", -0.1, 0.1) if paper else None,
            scale=p.NetworkScalarPrior("log-uniform", 0.01, 0.3)
            if profile == "p1-778f0d0"
            else None,
            importance=p.NetworkScalarPrior("log-uniform", 0.2, 2.0),
        )
        examples[f"Source noise placement: {profile} (explicit importance adaptation)"] = {
            "topology": "network-innovations",
            "source": {
                "topology": "network-authored",
                "task": asdict(task),
                "network": asdict(network),
                "root": None,
            },
            "statistics": asdict(stats),
            "plan": None,
        }
    authored = original["Standalone network: sparse + gated + skip (authored)"]
    for law in ("gaussian", "uniform", "student_t", "mixture", "ball"):
        exogenous = (
            p.ExogenousDistribution(geometry="ball")
            if law == "ball"
            else p.ExogenousDistribution(p.InputDistribution(distribution=law, df=4.5))
        )
        plan = p.NetworkInnovationPlan(
            tuple(
                p.NetworkInnovationStage(
                    (0.1,) * width,
                    (0.2,) * width,
                    exogenous=exogenous,
                    response=p.NoiseResponse(offset=0.5, strength=0.15),
                )
                for width in (2, 1)
            ),
            importance=(0.4, 2.0),
            importance_gates=True,
        )
        examples[f"Noisy authored combination: {law} signal-dependent extension"] = {
            "topology": "network-innovations",
            "source": authored,
            "statistics": None,
            "plan": asdict(plan),
        }
    return examples
