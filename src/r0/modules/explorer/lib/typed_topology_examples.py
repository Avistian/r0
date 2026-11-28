"""Editable typed-role topologies with complete prior records and explicit caps."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def typed_topology_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    linear = p.JointFunctionPrior("lin")
    for family in ("functional", "native", "native-vector", "converted"):
        request = p.TypedTopologyRequest(
            topology=p.FixedSizeTopologyHyperprior(4),
            task=p.LinearRecipe(
                rows=16, context_rows=8, features=2, signal_features=2, active_features=2
            ),
            family=family,
            width_prior=p.JointWidthPrior(1, 3),
            native=p.NativeTopologyRoles() if family.startswith("native") else None,
            converted=p.ConvertedTopologyRoles(
                (("low", "high"), ()),
                converter=p.CategoricalConverterPrior(
                    modes=("softmax_id",),
                    function=linear,
                    embeddings=p.JointRootPrior(function=linear),
                ),
            )
            if family == "converted"
            else None,
            pipeline=p.JointPipelinePrior(
                channel_importance=False,
                l2_normalization=False,
                node_importance=False,
            )
            if family == "converted"
            else None,
            calibration_rows=32,
            root_prior=p.JointRootPrior(function=linear),
            function_prior=linear,
            limits=p.CausalLimits(parameters=100_000_000, working_bytes=16 * 1024**3),
            raw_limits=p.RawDrawLimits(numbers=100_000_000, working_bytes=16 * 1024**3),
        )
        examples[f"Topology hyperprior: typed {family}"] = {
            "topology": "typed-topology-hyperprior",
            "request": asdict(request),
            "engine": "PCG64",
        }
    return examples
