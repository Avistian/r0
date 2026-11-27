"""Materialize a graph plan in an already staged shared construction scope."""

from dataclasses import replace

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_graph_admission import admit_joint_graph
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePlan
from r0.modules.priors.lib.joint_pipeline_realization import (
    JointPipelineRealization,
    fit_node_pipeline,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.materialized_joint_graph import MaterializedJointGraph
from r0.modules.priors.lib.materialized_parent_graph import MaterializedParentGraph
from r0.modules.priors.lib.parent_composition_plans import (
    ParentCompositionPrior,
    ParentFunctionPlan,
)
from r0.modules.priors.lib.parent_function_realization import (
    ParentCalibrationReceipt,
    ParentFunctionRealization,
    materialize_parent_function,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.sampled_functional_graph import (
    GraphCalibrationReceipt,
)


def _needs_fit(plan: JointFunctionPlan) -> bool:
    return plan.family in ("tree", "disc", "em", "mlp") or (
        plan.family == "prod" and any(_needs_fit(child) for child in plan.children)
    )


def materialize_joint_graph(
    request: JointGraphRequest,
    plan: JointGraphPlan,
    registry: JointOperationRegistry,
    staged: dict[str, np.random.Generator],
    initial: tuple[GeneratorSnapshot, ...],
    *,
    pipeline_plans: tuple[JointPipelinePlan, ...] = (),
    pipeline_records: list[JointPipelineRealization] | None = None,
    composition: ParentCompositionPrior | None = None,
) -> MaterializedJointGraph | MaterializedParentGraph:
    plan.validate(registry)
    if pipeline_plans and (len(pipeline_plans) != len(plan.nodes) or pipeline_records is None):
        raise ValueError("sampled pipeline plans require aligned construction records")
    admission = admit_joint_graph(request, plan, registry)
    ledger = RawDrawLedger(limits=admission.raw_limits)
    nodes: list[FunctionalCausalNode] = []
    realizations: list[
        JointRootRealization | RootFunctionRealization | ParentFunctionRealization
    ] = []
    receipts: list[GraphCalibrationReceipt] = []
    parent_receipts: list[ParentCalibrationReceipt] = []
    with ledger.transactions({name: staged[name] for name in ("raw", "calibration", "fit")}) as (
        streams,
        capture,
    ):
        for i, node in enumerate(plan.nodes):
            mechanism = node.mechanism
            identity = f"graph-node-{i}-function-fit"
            realization: JointRootRealization | RootFunctionRealization | ParentFunctionRealization
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
            elif isinstance(mechanism, ParentFunctionPlan):

                def parent_population(role: str, node_index: int = i):
                    incoming = plan.nodes[node_index].parents
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
                    values = sample_functional_prefix(
                        prefix,
                        rng=streams["calibration"],
                        owner=("graph", "construction", f"node:{node_index}", role),
                        capture=capture,
                    )
                    parent_receipts.append(
                        ParentCalibrationReceipt(
                            node_index, role, incoming, start, len(capture.freeze())
                        )
                    )
                    return torch.cat(tuple(values.states[p] for p in incoming), dim=1).numpy()

                realization = materialize_parent_function(
                    mechanism,
                    registry.decisions,
                    streams=streams,
                    capture=capture,
                    population=parent_population,
                    limits=request.limits,
                )
                function, root = realization.function, None
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
            if pipeline_plans:
                assert pipeline_records is not None
                # The current node contributes only its function output to the fit;
                # all preceding nodes already have final pipelines and noise.
                bare = replace(nodes[-1], pipeline=None, noise_scale=())
                fitted = fit_node_pipeline(
                    pipeline_plans[i],
                    registry.decisions,
                    (*nodes[:-1], bare),
                    rows=request.calibration_rows,
                    streams=streams,
                    capture=capture,
                    limits=request.limits,
                    retained=(
                        request,
                        plan,
                        registry,
                        tuple(realizations),
                        tuple(pipeline_records),
                        capture.freeze(),
                        initial,
                    ),
                )
                pipeline_records.append(fitted)
                nodes[-1] = replace(
                    nodes[-1],
                    pipeline=fitted.pipeline,
                    noise_scale=(fitted.plan.sigma,) * node.width,
                )
            realizations.append(realization)
        if pipeline_plans:
            request = replace(
                request,
                authored_pipelines=tuple(n.pipeline for n in nodes),
                noise=tuple(
                    JointGraphNoise(n.noise_scale, n.exogenous, n.noise_response) for n in nodes
                ),
            )
            # The enclosing pipeline request separately reserves retained fits and
            # transient buffers. Standard graph validation binds the final authored
            # equations and independently recomputes this final graph admission.
            final_admission = admit_joint_graph(request, plan, registry)
            capture.admit_limits(final_admission.raw_limits)
        else:
            final_admission = admission
        runtime = FunctionalCausalRecipe(
            request.task, tuple(nodes), request.observed, request.target, request.limits
        )
        raw = capture.freeze()
    if composition is not None:
        if receipts or any(
            not isinstance(r, JointRootRealization | ParentFunctionRealization)
            for r in realizations
        ):
            raise ValueError("parent construction cannot mix legacy nonroot realizations")
        return MaterializedParentGraph(
            request,
            plan,
            registry,
            plan.owners(),
            tuple(
                r
                for r in realizations
                if isinstance(r, JointRootRealization | ParentFunctionRealization)
            ),
            tuple(parent_receipts),
            raw,
            final_admission,
            runtime,
            composition,
        )
    if parent_receipts or any(isinstance(r, ParentFunctionRealization) for r in realizations):
        raise ValueError("parent construction requires its explicit composition prior")
    return MaterializedJointGraph(
        request,
        plan,
        registry,
        plan.owners(),
        tuple(r for r in realizations if not isinstance(r, ParentFunctionRealization)),
        tuple(receipts),
        raw,
        final_admission,
        runtime,
    )
