"""Complete public joint requests; catalogs are never narrowed by the explorer."""

import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def topology_hyperprior_examples() -> dict[str, Any]:
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    function = p.JointFunctionPrior("lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
    request = p.JointHyperpriorGraphRequest(
        task,
        p.FixedSizeTopologyHyperprior(3),
        calibration_rows=24,
        root_prior=p.JointRootPrior(("gaussian",), function),
        function_prior=function,
        limits=p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3),
    )
    authored = p.HyperpriorGraphRequest(
        task,
        p.FixedSizeTopologyHyperprior(3),
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        limits=request.limits,
    )
    examples: dict[str, Any] = {
        "Topology hyperprior: authored linear": {
            "topology": "authored-topology-hyperprior",
            "request": asdict(authored),
            "engine": "PCG64",
        },
        "Topology hyperprior: joint linear": {
            "topology": "joint-topology-hyperprior",
            "request": asdict(request),
            "engine": "PCG64",
        },
    }

    logarithmic = deepcopy(examples["Topology hyperprior: joint linear"])
    logarithmic["request"]["topology"] = asdict(
        p.TopologyHyperprior(
            log_nodes_low=math.log(3),
            log_nodes_high=math.log(4),
            probability_transform="gamma-exp-r0",
        )
    )
    examples["Topology hyperprior: joint log-size"] = logarithmic
    for token in (
        "lin",
        "quad",
        "gp",
        "tree",
        "disc",
        "em",
        "mlp",
        "prod",
        "kumo-prod",
        "default",
        "default,kumo-prod",
    ):
        # Keep the complete support. High conservative caps are visible in the
        # draft and require explicit adoption by the caller before construction.
        raw = asdict(request)
        raw["function_prior"] = asdict(p.JointFunctionPrior(token, frequencies=8))
        raw["root_prior"] = asdict(
            p.JointRootPrior(
                function=p.JointFunctionPrior("r0-linear-projection-v1", frequencies=8)
            )
        )
        raw["limits"] = asdict(p.CausalLimits(parameters=64_000_000, working_bytes=16 * 1024**3))
        raw["raw_limits"] = asdict(p.RawDrawLimits(numbers=16_000_000, working_bytes=4 * 1024**3))
        examples[f"Topology hyperprior: joint catalog {token}"] = {
            "topology": "joint-topology-hyperprior",
            "request": raw,
            "engine": "PCG64",
        }
    return examples
