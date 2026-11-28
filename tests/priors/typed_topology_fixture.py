"""Small public-API typed topology controls for composition checks."""

from typing import Any

from r0.modules import priors as p


def request(family: str) -> p.TypedTopologyRequest:
    linear = p.JointFunctionPrior(functions="lin")
    controls: dict[str, Any] = dict(
        topology=p.FixedSizeTopologyHyperprior(4),
        task=p.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        family=family,
        width_prior=p.JointWidthPrior(1, 3),
        calibration_rows=32,
        root_prior=p.JointRootPrior(function=linear),
        function_prior=linear,
        limits=p.CausalLimits(parameters=100_000_000, working_bytes=16 * 1024**3),
        raw_limits=p.RawDrawLimits(numbers=100_000_000, working_bytes=16 * 1024**3),
    )
    if family in ("native", "native-vector"):
        controls["native"] = p.NativeTopologyRoles()
    if family == "converted":
        controls["converted"] = p.ConvertedTopologyRoles(
            ((), ()),
            converter=p.CategoricalConverterPrior(
                modes=("softmax_id",), function=linear, embeddings=p.JointRootPrior(function=linear)
            ),
        )
        controls["pipeline"] = p.JointPipelinePrior(
            channel_importance=False, l2_normalization=False, node_importance=False
        )
    return p.TypedTopologyRequest(**controls)
