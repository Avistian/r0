"""One shared preference scope for native domains and all structural mechanisms."""

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_function_sampling import sample_joint_function
from r0.modules.priors.lib.joint_root_plans import JointRootPlan, sample_joint_root
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.native_domain_bindings import validate_native_domain
from r0.modules.priors.lib.native_domain_prior import sample_native_domain
from r0.modules.priors.lib.native_domain_realization import materialize_native_domain
from r0.modules.priors.lib.native_graph_admission import admit_native_construction
from r0.modules.priors.lib.native_vector_evaluation import evaluate_native_vector_nodes
from r0.modules.priors.lib.native_vector_records import (
    NativeVectorNode,
    NativeVectorRecipe,
    native_vector_certificates,
)
from r0.modules.priors.lib.native_vector_request import (
    NativeVectorDomainPlan,
    NativeVectorGraphRequest,
)
from r0.modules.priors.lib.parent_composition_plans import (
    ParentFunctionPlan,
    sample_parent_function,
)
from r0.modules.priors.lib.parent_function_realization import (
    ParentCalibrationReceipt,
    ParentFunctionRealization,
    materialize_parent_function,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.sampled_native_vector_graph import SampledNativeVectorGraph


def _needs_fit(plan: JointFunctionPlan) -> bool:
    return plan.family in ("tree", "disc", "em", "mlp") or any(
        _needs_fit(child) for child in plan.children
    )


def sample_native_vector_graph(
    request: NativeVectorGraphRequest,
    *,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledNativeVectorGraph:
    if type(request) is not NativeVectorGraphRequest:
        raise ValueError("native graph sampling requires its frozen request")
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
        domains: list[NativeVectorDomainPlan] = []
        mechanisms: list[JointRootPlan | JointFunctionPlan | ParentFunctionPlan] = []
        for i, parents in enumerate(request.parents):
            with context.transaction(f"node:{i}") as operation:
                domain = NativeVectorDomainPlan(
                    tuple(
                        sample_native_domain(operation, f"domain:{j}", prior)
                        for j, prior in enumerate(request.domains[i])
                    )
                )
                domains.append(domain)
                if parents and request.composition is not None:
                    mechanism = sample_parent_function(
                        operation,
                        "function",
                        parent_widths=tuple(domains[p].state_width for p in parents),
                        output_width=domain.score_width,
                        calibration_rows=request.calibration_rows,
                        functions=request.function_prior,
                        prior=request.composition,
                    )
                elif parents:
                    mechanism = sample_joint_function(
                        operation,
                        "function",
                        input_width=sum(domains[p].state_width for p in parents),
                        output_width=domain.score_width,
                        calibration_rows=request.calibration_rows,
                        prior=request.function_prior,
                    )
                else:
                    mechanism = sample_joint_root(
                        operation,
                        "root",
                        width=domain.score_width,
                        calibration_rows=request.calibration_rows,
                        prior=request.root_prior,
                    )
                mechanisms.append(mechanism)
        registry = context.freeze_registry()
        for domain in domains:
            for variable in domain.variables:
                validate_native_domain(variable, registry.decisions)
        raw_limits = admit_native_construction(request, tuple(domains), tuple(mechanisms), registry)
        ledger = RawDrawLedger(limits=raw_limits)
        nodes: list[NativeVectorNode] = []
        realizations: list[
            JointRootRealization | RootFunctionRealization | ParentFunctionRealization
        ] = []
        receipts: list[ParentCalibrationReceipt] = []
        with ledger.transactions({k: staged[k] for k in ("raw", "calibration", "fit")}) as (
            streams,
            capture,
        ):
            for i, mechanism in enumerate(mechanisms):
                domain = tuple(
                    materialize_native_domain(v, streams["raw"], capture)
                    for v in domains[i].variables
                )
                identity = f"native-node-{i}-function-fit"
                realization: (
                    JointRootRealization | RootFunctionRealization | ParentFunctionRealization
                )
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
                        native_vector_certificates(tuple(nodes))
                        start = len(capture.freeze())
                        values = evaluate_native_vector_nodes(
                            tuple(nodes),
                            request.calibration_rows,
                            lambda *keys: streams["calibration"],
                            "cpu",
                            capture=capture,
                            owner=("native", f"receiver:{node_index}", role),
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
                            [values.states[p] for p in request.parents[node_index]], dim=1
                        ).numpy()

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
                    if _needs_fit(mechanism):
                        native_vector_certificates(tuple(nodes))
                        population = evaluate_native_vector_nodes(
                            tuple(nodes),
                            request.calibration_rows,
                            lambda *keys: streams["calibration"],
                            "cpu",
                            capture=capture,
                            owner=("native", f"receiver:{i}", "calibration"),
                        )
                        calibration = torch.cat(
                            [population.states[p] for p in request.parents[i]], dim=1
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
                    function, root = realization.function, None
                nodes.append(
                    NativeVectorNode(
                        FunctionalCausalNode(request.parents[i], function, root), domain
                    )
                )
                realizations.append(realization)
            target = domains[request.target[0]].variables[request.target[1]]
            recipe = NativeVectorRecipe(
                request.task(kind=target.kind, cardinality=target.cardinality),
                tuple(nodes),
                request.observed,
                request.target,
                request.limits,
            )
            raw = capture.freeze()
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        return SampledNativeVectorGraph(
            request,
            tuple(domains),
            tuple(mechanisms),
            registry,
            tuple(realizations),
            raw,
            recipe,
            initial,
            final,
            raw_limits,
            calibrations=tuple(receipts),
        )
