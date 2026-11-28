"""One topology realization feeds existing typed/geometry samplers without new edges."""

from typing import Any

from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
from r0.modules.priors.lib.fixed_parent_topology import FixedParentTopology
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_parent_request import GeometryParentRequest
from r0.modules.priors.lib.geometry_pipeline_request import GeometryPipelineRequest
from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.native_parent_request import NativeParentGraphRequest
from r0.modules.priors.lib.native_vector_request import NativeVectorGraphRequest
from r0.modules.priors.lib.typed_topology_admission import typed_topology_child_limits
from r0.modules.priors.lib.typed_topology_geometry import NativeTopologyGeometry
from r0.modules.priors.lib.typed_topology_request import (
    TypedParentTopologyRequest,
    TypedTopologyRequest,
)


def typed_topology_child_request(
    request: TypedTopologyRequest,
    parents: tuple[tuple[int, ...], ...],
    geometry: NativeTopologyGeometry | None,
):
    composition = request.composition if isinstance(request, TypedParentTopologyRequest) else None
    limits = typed_topology_child_limits(request)
    common: dict[str, Any] = dict(
        calibration_rows=request.calibration_rows,
        root_prior=request.root_prior,
        function_prior=request.function_prior,
        mode=request.mode,
        categorical_policy=request.categorical_policy,
        global_policy=request.global_policy,
        limits=limits,
        raw_limits=request.raw_limits,
    )
    if request.family == "functional":
        if geometry is not None:
            raise ValueError("functional topology uses the retained child geometry")
        controls: dict[str, Any] = dict(
            topology=FixedParentTopology(parents),
            task=request.task,
            subsample=request.subsample,
            width_prior=request.width_prior,
            **common,
        )
        if composition is not None:
            return GeometryParentRequest(
                **controls, composition=composition, pipeline_prior=request.pipeline
            )
        return (
            GeometryGraphRequest(**controls)
            if request.pipeline is None
            else GeometryPipelineRequest(**controls, pipeline_prior=request.pipeline)
        )
    if request.family == "converted":
        if geometry is not None or request.converted is None or request.pipeline is None:
            raise ValueError("converted topology requires its child geometry and explicit priors")
        return ConvertedGraphRequest(
            FixedParentTopology(parents),
            request.converted.vocabularies,
            request.converted.target_vocabulary,
            request.task.rows,
            request.task.context_rows,
            request.task.split,
            request.subsample,
            request.width_prior,
            pipeline_prior=request.pipeline,
            converter_prior=request.converted.converter,
            composition=composition,
            **common,
        )
    if geometry is None or request.native is None:
        raise ValueError("native topology requires complete semantic geometry")
    geometry.validate(request, len(parents))
    features = {n for n, _ in geometry.observed}
    domains = tuple(
        request.native.domain(root=not row, feature=i in features, target=i == geometry.target[0])
        for i, row in enumerate(parents)
    )
    native_controls: dict[str, Any] = dict(
        parents=parents,
        observed=geometry.observed,
        target=geometry.target,
        rows=request.task.rows,
        context_rows=request.task.context_rows,
        split=request.task.split,
        **common,
    )
    if request.family == "native":
        constructor = NativeParentGraphRequest if composition is not None else NativeGraphRequest
        if composition is not None:
            native_controls["composition"] = composition
        return constructor(
            domains=tuple(
                request.native.for_width(d, w)
                for d, w in zip(domains, geometry.widths, strict=True)
            ),
            **native_controls,
        )
    # First variable carries the semantic node role; remaining variables are
    # genuinely hidden variables, each with its own sampled domain and output.
    return NativeVectorGraphRequest(
        composition=composition,
        domains=tuple(
            (d,) + (request.native.hidden,) * (w - 1)
            for d, w in zip(domains, geometry.widths, strict=True)
        ),
        **native_controls,
    )
