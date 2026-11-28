"""Metadata admission and exact child binding precede five-stream replay."""

from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.categorical_converter_plans import CategoricalConverterPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fixed_parent_topology import FixedParentTopology, RetainedParentTopology
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.geometry_topology import decode_geometry_topology
from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.joint_pipeline_decoding import pipeline_prior_from_dict
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_graph_decoding import native_domain_prior_from_dict
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.sampled_graph_decoding import decode_joint_function_prior
from r0.modules.priors.lib.topology_hyperprior_decoding import complete_topology_record as record
from r0.modules.priors.lib.topology_hyperprior_decoding import (
    sampled_topology_hyperprior_metadata,
    topology_hyperprior_from_dict,
)
from r0.modules.priors.lib.typed_topology_adapters import typed_topology_child_request
from r0.modules.priors.lib.typed_topology_geometry import NativeTopologyGeometry
from r0.modules.priors.lib.typed_topology_graph import (
    TypedTopologyGraph,
    sample_typed_topology_graph,
)
from r0.modules.priors.lib.typed_topology_request import (
    TypedParentTopologyRequest,
    TypedTopologyRequest,
)
from r0.modules.priors.lib.typed_topology_roles import ConvertedTopologyRoles, NativeTopologyRoles

_DEFAULT_LIMITS = CausalLimits()


def root_prior(data: Any) -> JointRootPrior:
    raw = record(data, JointRootPrior)
    raw["families"] = tuple(raw["families"])
    raw["function"] = decode_joint_function_prior(raw["function"])
    return JointRootPrior(**raw)


def typed_topology_request_from_dict(data: Any, *, limits: CausalLimits) -> TypedTopologyRequest:
    admit_function_metadata(data, limits)
    composed = (
        type(data) is dict
        and cast(dict[str, Any], data).get("version") == "typed-parent-topology-request-r0-v1"
    )
    kind = TypedParentTopologyRequest if composed else TypedTopologyRequest
    raw = record(data, kind)
    if composed:
        raw["composition"] = ParentCompositionPrior(
            **record(raw["composition"], ParentCompositionPrior)
        )
    declared = CausalLimits(**record(raw["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("typed topology declared limits exceed caller limits")
    raw["limits"] = declared
    raw["raw_limits"] = RawDrawLimits(**record(raw["raw_limits"], RawDrawLimits))
    raw["topology"] = (
        decode_geometry_topology(raw["topology"])
        if composed and raw["topology"].get("version") == "fixed-parent-topology-r0-v1"
        else topology_hyperprior_from_dict(raw["topology"])
    )
    task = record(raw["task"], LinearRecipe)
    task["class_offsets"] = tuple(task["class_offsets"])
    raw["task"] = LinearRecipe(**task)
    raw["width_prior"] = JointWidthPrior(**record(raw["width_prior"], JointWidthPrior))
    raw["root_prior"] = root_prior(raw["root_prior"])
    raw["function_prior"] = decode_joint_function_prior(raw["function_prior"])
    if raw["pipeline"] is not None:
        raw["pipeline"] = pipeline_prior_from_dict(raw["pipeline"])
    if raw["native"] is not None:
        roles = record(raw["native"], NativeTopologyRoles)
        for key in ("roots", "hidden", "features", "target"):
            roles[key] = native_domain_prior_from_dict(roles[key])
        raw["native"] = NativeTopologyRoles(**roles)
    if raw["converted"] is not None:
        roles = record(raw["converted"], ConvertedTopologyRoles)
        roles["vocabularies"] = tuple(tuple(v) for v in roles["vocabularies"])
        roles["target_vocabulary"] = tuple(roles["target_vocabulary"])
        prior = record(roles["converter"], CategoricalConverterPrior)
        prior["modes"] = tuple(prior["modes"])
        prior["function"] = decode_joint_function_prior(prior["function"])
        prior["embeddings"] = root_prior(prior["embeddings"])
        roles["converter"] = CategoricalConverterPrior(**prior)
        raw["converted"] = ConvertedTopologyRoles(**roles)
    return kind(**raw)


def native_geometry(data: Any) -> NativeTopologyGeometry:
    raw = record(data, NativeTopologyGeometry)
    raw["widths"] = tuple(raw["widths"])
    raw["observed"] = tuple(tuple(x) for x in raw["observed"])
    raw["target"] = tuple(raw["target"])
    draws: list[RawDrawRecord] = []
    for item in raw["raw"]:
        draw = record(item, RawDrawRecord)
        for key in ("owner", "shape", "values"):
            draw[key] = tuple(draw[key])
        draw["parameters"] = tuple(tuple(x) for x in draw["parameters"])
        draws.append(RawDrawRecord(**draw))
    raw["raw"] = tuple(draws)
    return NativeTopologyGeometry(**raw)


def typed_topology_graph_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> TypedTopologyGraph:
    if type(limits) is not CausalLimits:
        raise ValueError("typed topology decoder requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        raw = record(data, TypedTopologyGraph)
        if raw["version"] not in (
            "sampled-typed-topology-r0-v1",
            "sampled-typed-parent-topology-r0-v1",
        ):
            raise ValueError("unsupported typed topology envelope version")
        request = typed_topology_request_from_dict(raw["request"], limits=limits)
        admit_function_metadata(data, request.limits)
        composed = isinstance(request, TypedParentTopologyRequest)
        if (raw["version"] == "sampled-typed-parent-topology-r0-v1") != composed:
            raise ValueError("typed topology envelope and request versions differ")
        if isinstance(request.topology, FixedParentTopology):
            top_raw = record(raw["topology"], RetainedParentTopology)
            top_prior = decode_geometry_topology(top_raw["prior"])
            if type(top_prior) is not FixedParentTopology:
                raise ValueError("composed topology requires its retained fixed parents")
            topology = RetainedParentTopology(
                top_prior,
                CausalLimits(**record(top_raw["limits"], CausalLimits)),
                top_raw["version"],
            )
            topology.validate()
            if topology.prior != request.topology or topology.limits != request.limits:
                raise ValueError("fixed parent receipt differs from original request")
        else:
            topology = sampled_topology_hyperprior_metadata(raw["topology"], limits=request.limits)
            if (
                not gp_payload_equal(asdict(topology.request), asdict(request.topology))
                or topology.limits != request.limits
                or topology.raw_limits != request.raw_limits
            ):
                raise ValueError("typed topology receipt differs from original request")
        geometry = None if raw["geometry"] is None else native_geometry(raw["geometry"])
        child = typed_topology_child_request(request, topology.parents, geometry)
        if type(raw["graph"]) is not dict or not gp_payload_equal(
            cast(dict[str, Any], raw["graph"]).get("request"), asdict(child)
        ):
            raise ValueError("typed topology child request differs from retained geometry")
        snapshots: list[tuple[GeneratorSnapshot, ...]] = []
        for role in ("initial", "final"):
            if type(raw[role]) not in (list, tuple) or len(raw[role]) != 5:
                raise ValueError("typed topology requires five complete stream snapshots")
            states = tuple(GeneratorSnapshot(**record(s, GeneratorSnapshot)) for s in raw[role])
            for state in states:
                state.validate_metadata(byte_limit=request.raw_limits.state_bytes)
            snapshots.append(states)
        if isinstance(topology, RetainedParentTopology):
            if snapshots[0][0] != snapshots[1][0]:
                raise ValueError("retained fixed topology must not consume a topology stream")
        elif snapshots[0][0] != topology.initial or snapshots[1][0] != topology.final:
            raise ValueError("typed topology outer stream differs from retained topology")
        initial, final = snapshots
        for a, b in zip(initial, final, strict=True):
            if a.engine != b.engine:
                raise ValueError("typed topology stream engines changed")
        child_record = cast(dict[str, Any], raw["graph"])
        versions = {
            "functional": "sampled-geometry-graph-v1",
            "native": "sampled-native-graph-v1",
            "native-vector": "sampled-native-vector-graph-v1",
            "converted": "sampled-converted-graph-v1",
        }
        if child_record.get("version") != versions[request.family]:
            raise ValueError("typed topology mechanism family differs")
        for role, states in (("initial", initial), ("final", final)):
            child_states = child_record.get(role)
            if type(child_states) not in (tuple, list):
                raise ValueError("typed topology child needs four stream snapshots")
            child_states = cast(list[Any] | tuple[Any, ...], child_states)
            if len(child_states) != 4:
                raise ValueError("typed topology child needs four stream snapshots")
            child_states = tuple(
                GeneratorSnapshot(**record(s, GeneratorSnapshot)) for s in child_states
            )
            for i, state in enumerate(child_states):
                state.validate_metadata(byte_limit=request.raw_limits.state_bytes)
                if role == "initial" and i == 1 and geometry is not None:
                    if (
                        state.engine != states[2].engine
                        or state.digest != geometry.raw[-1].state_after
                        or geometry.raw[0].state_before != states[2].digest
                    ):
                        raise ValueError(
                            "native geometry does not connect original and mechanism raw streams"
                        )
                elif state != states[i + 1]:
                    raise ValueError("typed topology child stream differs from its outer owner")
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in initial)
        replay = sample_typed_topology_graph(
            request,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
        if not gp_payload_equal(asdict(replay), data):
            raise ValueError("typed topology archive differs from complete replay")
        return replay
    except (
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ValueError("malformed typed topology archive") from error
