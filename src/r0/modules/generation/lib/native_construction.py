"""Translate sampled native transport records through public prior interfaces."""

from typing import Any

import numpy as np

from r0.modules import priors


def construct_native(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.SampledNativeGraph:
    def function(raw: dict[str, Any]) -> priors.JointFunctionPrior:
        fields = dict(raw)
        fields["matrix"] = priors.JointMatrixPrior(**fields["matrix"])
        return priors.JointFunctionPrior(**fields)

    fields = dict(fields)
    fields["domains"] = tuple(priors.native_domain_prior_from_dict(d) for d in fields["domains"])
    fields["function_prior"] = function(fields["function_prior"])
    root: dict[str, Any] = dict(fields["root_prior"])
    root["function"] = function(root["function"])
    fields["root_prior"] = priors.JointRootPrior(**root)
    fields["raw_limits"] = priors.RawDrawLimits(**fields["raw_limits"])
    request_type = priors.NativeGraphRequest
    if "composition" in fields:
        fields["composition"] = priors.ParentCompositionPrior(**fields["composition"])
        request_type = priors.NativeParentGraphRequest
    request = request_type(limits=limits, **fields)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    return priors.sample_native_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
