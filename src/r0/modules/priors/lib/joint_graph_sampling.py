"""Assemble named source plans into frozen graphs using four atomic caller streams."""

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_graph_admission import admit_joint_graph
from r0.modules.priors.lib.joint_graph_plans import sample_joint_graph_plan
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.sampled_functional_graph import (
    GraphCalibrationReceipt,
    SampledFunctionalGraph,
)


def _needs_fit(plan: JointFunctionPlan) -> bool:
    return plan.family in ("tree", "disc", "em", "mlp") or (
        plan.family == "prod" and any(_needs_fit(child) for child in plan.children)
    )


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
        admission = admit_joint_graph(request, plan, registry)
        ledger = RawDrawLedger(limits=admission.raw_limits)
        nodes: list[FunctionalCausalNode] = []
        realizations: list[JointRootRealization | RootFunctionRealization] = []
        receipts: list[GraphCalibrationReceipt] = []
        with ledger.transactions(
            {name: staged[name] for name in ("raw", "calibration", "fit")}
        ) as (streams, capture):
            for i, node in enumerate(plan.nodes):
                mechanism = node.mechanism
                identity = f"graph-node-{i}-function-fit"
                realization: JointRootRealization | RootFunctionRealization
                if isinstance(mechanism, JointRootPlan):
                    realization = materialize_joint_root(
                        mechanism,
                        registry.decisions,
                        rng=streams["raw"],
                        calibration_rng=streams["calibration"],
                        fit_rng=streams["fit"],
                        calibration_id=identity,
                        limits=request.limits,
                        capture=capture,
                    )
                    function, root = realization.function.function, realization.calibration.base
                else:
                    calibration = None
                    start = stop = 0
                    if _needs_fit(mechanism):
                        prefix = FunctionalPrefix(
                            tuple(nodes),
                            request.calibration_rows,
                            request.limits,
                            retained=(
                                request,
                                plan,
                                registry,
                                tuple(realizations),
                                capture.freeze(),
                                initial,
                            ),
                        )
                        start = len(capture.freeze())
                        population = sample_functional_prefix(
                            prefix,
                            rng=streams["calibration"],
                            owner=("graph", "construction", f"node:{i}", "function-fit"),
                            capture=capture,
                        )
                        stop = len(capture.freeze())
                        calibration = torch.cat(
                            tuple(population.states[p] for p in node.parents), dim=1
                        ).numpy()
                    realization = materialize_joint_function(
                        mechanism,
                        registry.decisions,
                        rng=streams["raw"],
                        fit_rng=streams["fit"],
                        calibration=calibration,
                        calibration_id=identity,
                        limits=request.limits,
                        capture=capture,
                    )
                    if calibration is not None:
                        # The fitted realization already owns the exact matrix;
                        # the receipt refers to it without storing a second copy.
                        sample = getattr(realization, "calibration", None)
                        if sample is None:
                            raise ValueError("fitted graph function lacks its calibration record")
                        receipts.append(
                            GraphCalibrationReceipt(
                                i, identity, node.parents, start, stop, sample.digest
                            )
                        )
                    function, root = realization.function, None
                noise = request.noise[i] if request.noise else JointGraphNoise()
                pipeline = request.authored_pipelines[i] if request.authored_pipelines else None
                nodes.append(
                    FunctionalCausalNode(
                        node.parents,
                        function,
                        root,
                        noise.scales,
                        noise.exogenous,
                        noise.response,
                        pipeline,
                    )
                )
                realizations.append(realization)
            runtime = FunctionalCausalRecipe(
                request.task, tuple(nodes), request.observed, request.target, request.limits
            )
            raw = capture.freeze()
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        result = SampledFunctionalGraph(
            request,
            plan,
            registry,
            plan.owners(),
            tuple(realizations),
            tuple(receipts),
            raw,
            initial,
            final,
            admission,
            runtime,
        )
        result.validate_structure()
        return result
