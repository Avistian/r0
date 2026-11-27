"""Atomic shared-context sampling and independent ancestral fitting of parent graphs."""

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_root_plans import JointRootPlan, sample_joint_root
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.parent_composition_plans import (
    ParentCompositionPrior,
    ParentFunctionPlan,
    sample_parent_function,
)
from r0.modules.priors.lib.parent_function_realization import (
    ParentFunctionRealization,
    materialize_parent_function,
)
from r0.modules.priors.lib.parent_graph_admission import admit_parent_graph
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.sampled_parent_graph import ParentCalibrationReceipt, SampledParentGraph

_DEFAULT_COMPOSITION = ParentCompositionPrior()


def sample_parent_graph(
    request: JointGraphRequest,
    *,
    composition: ParentCompositionPrior = _DEFAULT_COMPOSITION,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledParentGraph:
    if type(request) is not JointGraphRequest or type(composition) is not ParentCompositionPrior:
        raise ValueError("parent graph sampling requires frozen request and composition prior")
    callers = {"named": named_rng, "raw": raw_rng, "calibration": calibration_rng, "fit": fit_rng}
    with raw_transactions(callers, None) as (staged, _):
        initial = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in staged.values()
        )
        context = JointSamplingContext(
            staged["named"],
            mode=request.mode,
            categorical_policy=request.categorical_policy,
            global_policy=request.global_policy,
        )
        plans: list[JointRootPlan | ParentFunctionPlan] = []
        owners = [JointPlanOwner(("graph",), "parent-graph", ())]
        with context.transaction("graph") as graph:
            for i, parents in enumerate(request.parents):
                with graph.transaction(f"node:{i}") as node:
                    plan = (
                        sample_parent_function(
                            node,
                            "mechanism",
                            parent_widths=tuple(request.widths[p] for p in parents),
                            output_width=request.widths[i],
                            calibration_rows=request.calibration_rows,
                            functions=request.function_prior,
                            prior=composition,
                        )
                        if parents
                        else sample_joint_root(
                            node,
                            "mechanism",
                            width=request.widths[i],
                            calibration_rows=request.calibration_rows,
                            prior=request.root_prior,
                        )
                    )
                    plans.append(plan)
                    owners.append(JointPlanOwner(node.operation, "node", ("plans", i)))
                    owners.extend(
                        plan.owners(("plans", i))
                        if isinstance(plan, ParentFunctionPlan)
                        else plan_owners(plan, ("plans", i))
                    )
        registry = context.freeze_registry()
        admission = admit_parent_graph(request, tuple(plans), registry)
        ledger = RawDrawLedger(limits=admission.raw_limits)
        nodes: list[FunctionalCausalNode] = []
        realized: list[JointRootRealization | ParentFunctionRealization] = []
        receipts: list[ParentCalibrationReceipt] = []
        with ledger.transactions({k: staged[k] for k in ("raw", "calibration", "fit")}) as (
            streams,
            capture,
        ):
            for i, plan in enumerate(plans):
                root = None
                realization: JointRootRealization | ParentFunctionRealization
                if isinstance(plan, JointRootPlan):
                    realization = materialize_joint_root(
                        plan,
                        registry.decisions,
                        rng=streams["raw"],
                        calibration_rng=streams["calibration"],
                        fit_rng=streams["fit"],
                        calibration_id=f"graph-node-{i}-root-fit",
                        limits=request.limits,
                        capture=capture,
                    )
                    function, root = realization.function.function, realization.calibration.base
                else:

                    def population(role: str, node_index: int = i):
                        prefix = FunctionalPrefix(
                            tuple(nodes),
                            request.calibration_rows,
                            request.limits,
                            retained=(
                                request,
                                tuple(plans),
                                registry,
                                tuple(realized),
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
                        receipts.append(
                            ParentCalibrationReceipt(
                                node_index,
                                role,
                                request.parents[node_index],
                                start,
                                len(capture.freeze()),
                            )
                        )
                        return torch.cat(
                            tuple(values.states[p] for p in request.parents[node_index]), dim=1
                        ).numpy()

                    realization = materialize_parent_function(
                        plan,
                        registry.decisions,
                        streams=streams,
                        capture=capture,
                        population=population,
                        limits=request.limits,
                    )
                    function = realization.function
                noise = request.noise[i] if request.noise else JointGraphNoise()
                pipeline = request.authored_pipelines[i] if request.authored_pipelines else None
                nodes.append(
                    FunctionalCausalNode(
                        request.parents[i],
                        function,
                        root,
                        noise.scales,
                        noise.exogenous,
                        noise.response,
                        pipeline,
                    )
                )
                realized.append(realization)
            runtime = FunctionalCausalRecipe(
                request.task, tuple(nodes), request.observed, request.target, request.limits
            )
            raw = capture.freeze()
        final = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in staged.values()
        )
        result = SampledParentGraph(
            request,
            composition,
            tuple(plans),
            registry,
            tuple(owners),
            tuple(realized),
            tuple(receipts),
            raw,
            initial,
            final,
            admission,
            runtime,
        )
        result.validate_structure()
        return result
