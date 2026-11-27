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
