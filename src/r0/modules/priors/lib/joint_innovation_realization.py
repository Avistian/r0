"""Frozen innovation geometry uses one raw factor with declared row-vector orientation."""

import numpy as np

from r0.modules.priors.lib.exogenous import ExogenousDistribution
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise
from r0.modules.priors.lib.joint_innovation_plans import JointInnovationPlan
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw


def materialize_innovation(
    plan: JointInnovationPlan, rng: np.random.Generator, capture: RawDrawLedger
) -> JointGraphNoise:
    factor: tuple[tuple[float, ...], ...] = ()
    if plan.factor == "gaussian":
        draw = recorded_draw(
            rng,
            capture,
            owner=plan.operation,
            role="innovation-factor",
            distribution="normal",
            shape=(plan.width, plan.width),
        ) / np.sqrt(plan.width)
        factor = tuple(tuple(float(v) for v in row) for row in draw)
    return JointGraphNoise(
        plan.scales,
        ExogenousDistribution(
            plan.inputs, factor=factor, geometry="ball" if plan.family == "ball" else "independent"
        ),
        plan.response,
    )
