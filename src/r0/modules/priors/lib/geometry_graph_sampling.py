"""Sample observation geometry and mechanisms without resetting named preferences."""

import numpy as np

from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_pipeline_request import GeometryPipelineRequest
from r0.modules.priors.lib.geometry_topology import sample_geometry_topology
from r0.modules.priors.lib.joint_graph_materialization import materialize_joint_graph
from r0.modules.priors.lib.joint_graph_plans import sample_joint_graph_plan
from r0.modules.priors.lib.joint_observation_plans import sample_joint_observations
from r0.modules.priors.lib.joint_observation_realization import materialize_joint_observations
from r0.modules.priors.lib.joint_pipeline_plans import sample_joint_pipeline
from r0.modules.priors.lib.joint_pipeline_realization import JointPipelineRealization
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_width_plans import sample_joint_widths
from r0.modules.priors.lib.observation_channel_layout import allocate_observation_channels
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_pipeline_geometry import SampledPipelineGeometry


def sample_geometry_graph(
    request: GeometryGraphRequest,
    *,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledGeometryGraph:
    if type(request) not in (GeometryGraphRequest, GeometryPipelineRequest):
        raise ValueError("geometry sampling requires its frozen request")
    request.mechanism_raw_limits()
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
        topology = sample_geometry_topology(
            request.topology,
            rng=staged["raw"],
            limits=request.limits,
            raw_limits=request.raw_limits,
        )
        observation_plan = sample_joint_observations(
            context,
            "observations",
            nodes=request.topology.nodes,
            groups=request.groups,
            subsample=request.subsample,
            limits=request.limits,
        )
        observations = materialize_joint_observations(
            observation_plan,
            context.freeze(),
            rng=staged["raw"],
            raw_limits=request.raw_limits,
        )
        layout = allocate_observation_channels(
            observations,
            context.freeze(),
            converter_widths=tuple((1,) * g.count for g in request.groups),
        )
        widths = sample_joint_widths(
            context,
            "widths",
            converter_channels=layout.converter_channels,
            prior=request.width_prior,
            limits=request.limits,
        )
        child_limits = request.mechanism_limits((topology, observations, layout, widths, initial))
        graph_request = request.graph_request(topology.parents, layout, widths, limits=child_limits)
        plan = sample_joint_graph_plan(
            context,
            "graph",
            parents=topology.parents,
            widths=widths.widths,
            calibration_rows=request.calibration_rows,
            root_prior=request.root_prior,
            function_prior=request.function_prior,
            limits=child_limits,
        )
        pipeline_plans = ()
        pipeline_records: list[JointPipelineRealization] = []
        if type(request) is GeometryPipelineRequest:
            converter_groups = [[""] * count for count in layout.converter_channels]
            for group, spans in zip(request.groups, layout.spans, strict=True):
                for node, start, stop in spans:
                    converter_groups[node][start:stop] = [group.name] * (stop - start)
            pipeline_plans = tuple(
                sample_joint_pipeline(
                    context,
                    f"pipeline:{i}",
                    width=width,
                    groups=tuple(converter_groups[i]),
                    prior=request.pipeline_prior,
                )
                for i, width in enumerate(widths.widths)
            )
        registry = context.freeze_registry()
        graph = materialize_joint_graph(
            graph_request,
            plan,
            registry,
            staged,
            initial,
            pipeline_plans=pipeline_plans,
            pipeline_records=pipeline_records,
        )
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in staged.values()
        )
        result = (
            SampledPipelineGeometry(
                request,
                topology,
                observations,
                layout,
                widths,
                graph,
                initial,
                final,
                pipelines=tuple(pipeline_records),
            )
            if type(request) is GeometryPipelineRequest
            else SampledGeometryGraph(
                request, topology, observations, layout, widths, graph, initial, final
            )
        )
        result.validate_structure()
        return result
