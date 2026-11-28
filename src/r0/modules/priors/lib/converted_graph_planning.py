"""Typed converters choose widths before shared geometry and mechanism plans."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.categorical_converter_plans import (
    CategoricalConverterPlan,
    sample_categorical_converter,
)
from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
from r0.modules.priors.lib.geometry_topology import GeometryTopology, sample_geometry_topology
from r0.modules.priors.lib.joint_converter_plans import JointConverterPlan, sample_joint_converter
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan, sample_joint_graph_plan
from r0.modules.priors.lib.joint_observation_plans import sample_joint_observations
from r0.modules.priors.lib.joint_observation_realization import (
    JointObservationRealization,
    materialize_joint_observations,
)
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePlan, sample_joint_pipeline
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_width_plans import JointWidthPlan, sample_joint_widths
from r0.modules.priors.lib.observation_channel_layout import (
    ObservationChannelLayout,
    allocate_observation_channels,
)


@dataclass(frozen=True, slots=True)
class ConvertedGraphPlan:
    categories: tuple[CategoricalConverterPlan | None, ...]
    topology: GeometryTopology
    observations: JointObservationRealization
    layout: ObservationChannelLayout
    widths: JointWidthPlan
    mechanisms: JointGraphPlan
    pipelines: tuple[JointPipelinePlan, ...]
    numerical: tuple[tuple[JointConverterPlan, ...], ...]
    version: str = "converted-graph-plan-v1"


def plan_converted_graph(
    request: ConvertedGraphRequest, context: JointSamplingContext, raw_rng: np.random.Generator
) -> ConvertedGraphPlan:
    geometry = request.geometry()
    geometry.mechanism_raw_limits()
    categories = tuple(
        sample_categorical_converter(
            context,
            f"category:{i}",
            vocabulary=vocabulary,
            calibration_rows=request.calibration_rows,
            prior=request.converter_prior,
        )
        if vocabulary
        else None
        for i, vocabulary in enumerate((*request.vocabularies, request.target_vocabulary))
    )
    topology = sample_geometry_topology(
        request.topology, rng=raw_rng, limits=request.limits, raw_limits=request.raw_limits
    )
    observation_plan = sample_joint_observations(
        context,
        "observations",
        nodes=request.topology.nodes,
        groups=geometry.groups,
        subsample=request.subsample,
        limits=request.limits,
    )
    observations = materialize_joint_observations(
        observation_plan, context.freeze(), rng=raw_rng, raw_limits=request.raw_limits
    )
    converter_widths = tuple(c.width if c is not None else 1 for c in categories)
    layout = allocate_observation_channels(
        observations,
        context.freeze(),
        converter_widths=(converter_widths[:-1], converter_widths[-1:]),
    )
    widths = sample_joint_widths(
        context,
        "widths",
        converter_channels=layout.converter_channels,
        prior=request.width_prior,
        limits=request.limits,
    )
    mechanisms = sample_joint_graph_plan(
        context,
        "graph",
        parents=topology.parents,
        widths=widths.widths,
        calibration_rows=request.calibration_rows,
        root_prior=request.root_prior,
        function_prior=request.function_prior,
        composition=request.composition,
        limits=request.limits,
    )
    pipelines = tuple(
        sample_joint_pipeline(
            context, f"pipeline:{i}", width=width, groups=(), prior=request.pipeline_prior
        )
        for i, width in enumerate(widths.widths)
    )
    numerical: list[list[JointConverterPlan]] = [[] for _ in widths.widths]
    for index, (category, (node, start, _)) in enumerate(
        zip(categories, (*layout.spans[0], *layout.spans[1]), strict=True)
    ):
        if category is None:
            numerical[node].append(
                sample_joint_converter(
                    context,
                    f"numerical:{index}",
                    channel=start,
                    feature_group="y" if index == len(categories) - 1 else "x",
                    prior=request.pipeline_prior.converter,
                )
            )
    return ConvertedGraphPlan(
        categories,
        topology,
        observations,
        layout,
        widths,
        mechanisms,
        pipelines,
        tuple(tuple(plans) for plans in numerical),
    )
