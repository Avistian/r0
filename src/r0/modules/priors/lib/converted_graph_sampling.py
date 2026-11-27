"""Materialize typed ancestral pipelines atomically, with independent fit populations."""

from dataclasses import replace

import numpy as np
import torch

from r0.modules.priors.lib.categorical_converter import CategoricalConverter
from r0.modules.priors.lib.categorical_converter_realization import (
    CategoricalConverterRealization,
    fit_categorical_converter,
)
from r0.modules.priors.lib.converted_causal_records import (
    ConvertedCausalNode,
    ConvertedCausalRecipe,
    ConvertedReadout,
    converted_certificates,
)
from r0.modules.priors.lib.converted_evaluation import evaluate_converted_nodes
from r0.modules.priors.lib.converted_graph_admission import admit_converted_construction
from r0.modules.priors.lib.converted_graph_planning import plan_converted_graph
from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_pipeline_realization import (
    JointPipelineRealization,
    fit_numerical_converters,
    fit_pipeline_population,
    pipeline_noise,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.parent_function_realization import (
    ParentCalibrationReceipt,
    ParentFunctionRealization,
    materialize_parent_function,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.sampled_converted_graph import SampledConvertedGraph


def needs_fit(plan: JointFunctionPlan) -> bool:
    return plan.family in ("tree", "disc", "em", "mlp") or any(needs_fit(c) for c in plan.children)


def sample_converted_graph(
    request: ConvertedGraphRequest,
    *,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledConvertedGraph:
    if type(request) is not ConvertedGraphRequest:
        raise ValueError("converted sampling requires its frozen request")
    with raw_transactions(
        {"named": named_rng, "raw": raw_rng, "calibration": calibration_rng, "fit": fit_rng}, None
    ) as (staged, _):
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
        plan = plan_converted_graph(request, context, staged["raw"])
        registry = context.freeze_registry()
        plan.mechanisms.validate(registry)
        for category in plan.categories:
            if category is not None:
                category.validate(registry.decisions)
        raw_limits = admit_converted_construction(request, plan, registry)
        ledger = RawDrawLedger(limits=raw_limits)
        nodes: list[ConvertedCausalNode] = []
        realizations: list[
            JointRootRealization | RootFunctionRealization | ParentFunctionRealization
        ] = []
        pipelines: list[JointPipelineRealization] = []
        converters: list[CategoricalConverterRealization] = []
        receipts: list[ParentCalibrationReceipt] = []
        spans = (*plan.layout.spans[0], *plan.layout.spans[1])
        with ledger.transactions({k: staged[k] for k in ("raw", "calibration", "fit")}) as (
            streams,
            capture,
        ):
            for i, node_plan in enumerate(plan.mechanisms.nodes):
                mechanism = node_plan.mechanism
                identity = f"converted-node-{i}-fit"

                def population(
                    role: str, index: int = i, parents: tuple[int, ...] = node_plan.parents
                ) -> torch.Tensor:
                    prefix = tuple(nodes)
                    converted_certificates(prefix)
                    admit_functional_graph(
                        tuple(n.equation for n in prefix),
                        request.limits,
                        rows=request.calibration_rows,
                        metadata=(
                            request,
                            plan,
                            registry,
                            prefix,
                            tuple(realizations),
                            tuple(pipelines),
                            tuple(converters),
                            capture.freeze(),
                        ),
                    )
                    start = len(capture.freeze())
                    values = evaluate_converted_nodes(
                        prefix,
                        request.calibration_rows,
                        lambda *keys: streams["calibration"],
                        "cpu",
                        capture=capture,
                        owner=("converted", f"receiver:{index}", role),
                    )
                    receipts.append(
                        ParentCalibrationReceipt(index, role, parents, start, len(capture.freeze()))
                    )
                    return (
                        values.states[-1]
                        if role == "pipeline"
                        else torch.cat([values.states[p] for p in parents], dim=1)
                    )

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
                    realization = materialize_parent_function(
                        mechanism,
                        registry.decisions,
                        streams=streams,
                        capture=capture,
                        population=lambda role: population(role).numpy(),
                        limits=request.limits,
                    )
                    function, root = realization.function, None
                else:
                    calibration = population("function").numpy() if needs_fit(mechanism) else None
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
                bare = FunctionalCausalNode(node_plan.parents, function, root)
                nodes.append(ConvertedCausalNode(bare))
                certificates = converted_certificates(tuple(nodes))[-1]
                if (
                    request.pipeline_prior.channel_importance
                    or request.pipeline_prior.l2_normalization
                ) and any(c.order <= 2 for c in certificates):
                    raise ValueError(
                        "pipeline fit requires certified finite second moments before sampling"
                    )
                start = len(capture.freeze())
                values = population("pipeline")
                fitted = fit_pipeline_population(
                    plan.pipelines[i],
                    registry.decisions,
                    values,
                    identity=identity + "-pipeline",
                    streams=streams,
                    capture=capture,
                    limits=request.limits,
                    raw_start=start,
                )
                preprocessed = PipelineEvaluation(fitted.pipeline, i, None).before_noise(values)
                numerical = fit_numerical_converters(
                    plan.numerical[i],
                    preprocessed,
                    identity=identity + "-numerical",
                    streams=streams,
                    capture=capture,
                    limits=request.limits,
                )
                node_converters: list[CategoricalConverter] = []
                for j, (category, (receiver, first, stop)) in enumerate(
                    zip(plan.categories, spans, strict=True)
                ):
                    if category is not None and receiver == i:
                        if (
                            category.mode.startswith("softmax_")
                            and not (
                                request.pipeline_prior.channel_importance
                                or request.pipeline_prior.l2_normalization
                            )
                            and any(c.order <= 2 for c in certificates[first:stop])
                        ):
                            raise ValueError("softmax fit requires certified finite second moments")
                        fitted_category = fit_categorical_converter(
                            category,
                            registry.decisions,
                            name=f"variable-{j}",
                            start=first,
                            values=preprocessed[:, first:stop].numpy(),
                            identity=identity + f"-category-{j}",
                            streams=streams,
                            capture=capture,
                            limits=request.limits,
                        )
                        converters.append(fitted_category)
                        node_converters.append(fitted_category.converter)
                noise = pipeline_noise(fitted)
                equation = replace(
                    bare,
                    pipeline=replace(fitted.pipeline, converters=numerical),
                    noise_scale=noise.scales,
                    exogenous=noise.exogenous,
                    noise_response=noise.response,
                )
                nodes[-1] = ConvertedCausalNode(equation, tuple(node_converters))
                converted_certificates(tuple(nodes))
                realizations.append(realization)
                pipelines.append(fitted)
            refs = tuple(
                ConvertedReadout(node, channel=first)
                if plan.categories[j] is None
                else ConvertedReadout(node, capture=f"variable-{j}")
                for j, (node, first, _) in enumerate(spans)
            )
            recipe = ConvertedCausalRecipe(
                request.task(), tuple(nodes), refs[:-1], refs[-1], request.limits
            )
            raw = capture.freeze()
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        return SampledConvertedGraph(
            request,
            plan,
            registry,
            tuple(realizations),
            tuple(pipelines),
            tuple(converters),
            tuple(receipts),
            raw,
            recipe,
            initial,
            final,
            raw_limits,
        )
