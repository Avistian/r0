"""Assemble named source plans into frozen graphs using four atomic caller streams."""

import numpy as np

from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_graph_materialization import materialize_joint_graph
from r0.modules.priors.lib.joint_graph_plans import sample_joint_graph_plan
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph


def sample_joint_graph(
    request: JointGraphRequest,
    *,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledFunctionalGraph:
    """Freeze a supplied DAG; every fitted receiver draws a new ancestor population.

    Topology, widths, readouts, noise and authored pipelines are explicit inputs
    in v1. Mechanisms are sampled. No caller state publishes until every graph
    constructor and aggregate validation has succeeded on private staged streams.
    """
    if type(request) is not JointGraphRequest:
        raise ValueError("graph sampling requires a frozen request")
    callers = {"named": named_rng, "raw": raw_rng, "calibration": calibration_rng, "fit": fit_rng}
    with raw_transactions(callers, None) as (staged, _):
        initial = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        context = JointSamplingContext(
            staged["named"],
            mode=request.mode,
            categorical_policy=request.categorical_policy,
            global_policy=request.global_policy,
        )
        plan = sample_joint_graph_plan(
            context,
            "graph",
            parents=request.parents,
            widths=request.widths,
            calibration_rows=request.calibration_rows,
            root_prior=request.root_prior,
            function_prior=request.function_prior,
            limits=request.limits,
        )
        registry = context.freeze_registry()
        plan.validate(registry)
        construction = materialize_joint_graph(request, plan, registry, staged, initial)
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        result = construction.explicit_graph(initial, final)
        result.validate_structure()
        return result
