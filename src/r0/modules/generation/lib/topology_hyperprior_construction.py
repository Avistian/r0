"""Full topology requests are admitted before constructing any of five streams."""

from typing import Any

import numpy as np

from r0.modules import priors as p


def construct_topology_hyperprior(
    fields: dict[str, Any], *, seed: int, limits: p.CausalLimits, authored: bool = False
) -> p.JointHyperpriorGraph | p.HyperpriorGraph:
    if set(fields) != {"request", "engine"}:
        raise ValueError("joint topology construction requires its full request and engine")
    if fields["engine"] not in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"):
        raise ValueError("unsupported topology construction engine")
    if authored:
        authored_request = p.hyperprior_graph_request_from_dict(fields["request"], limits=limits)
        rng = np.random.Generator(getattr(np.random, fields["engine"])(seed))
        return p.sample_hyperprior_graph(authored_request, rng=rng)
    request = p.joint_hyperprior_request_from_dict(fields["request"], limits=limits)
    streams = {
        name: np.random.Generator(getattr(np.random, fields["engine"])(child))
        for name, child in zip(
            ("topology_rng", "named_rng", "raw_rng", "calibration_rng", "fit_rng"),
            np.random.SeedSequence(seed).spawn(5),
            strict=True,
        )
    }
    return p.sample_joint_hyperprior_graph(request, **streams)
