"""Retained sampled kernel replay costs are additional to graph emission storage."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D


def convolution_replay_cost(function: object, limits: CausalLimits) -> tuple[int, int]:
    from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction

    members = (
        (*function.children, *((function.joint,) if function.joint else ()))
        if isinstance(function, FrozenParentFunction)
        else (function,)
    )
    parameters, work = 0, 0
    for member in members:
        if isinstance(member, SampledFeatureConv1D):
            report = member.request.admit(limits=limits)
            parameters += report["parameters"]
            work += report["estimated_working_bytes"]
    return parameters, work


def convolution_record_adjustment(function: object, limits: CausalLimits) -> tuple[int | None, int]:
    """Charge complete convolution or parent metadata instead of coefficient count."""
    from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
    from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
    from r0.modules.priors.lib.joint_function_admission import function_record_cost

    members = (
        (*function.children, *((function.joint,) if function.joint else ()))
        if isinstance(function, FrozenParentFunction)
        else (function,)
    )
    if any(isinstance(member, (FrozenFeatureConv1D, SampledFeatureConv1D)) for member in members):
        parameters, storage = function_record_cost(function, limits)
        return parameters, storage - 64 * parameters
    return None, 0
