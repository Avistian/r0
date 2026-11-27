"""Exact outer/request bindings and whole support admission before all nested replay."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.hyperprior_graph_admission import hyperprior_readouts
from r0.modules.priors.lib.joint_hyperprior_graph import JointHyperpriorGraph
from r0.modules.priors.lib.joint_hyperprior_request import JointHyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_graph_request,
    sampled_graph_from_dict,
)
from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior
from r0.modules.priors.lib.topology_hyperprior_decoding import (
    complete_topology_record,
    topology_hyperprior_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def joint_hyperprior_request_from_dict(
    data: Any, *, limits: CausalLimits
) -> JointHyperpriorGraphRequest:
    if type(limits) is not CausalLimits:
        raise ValueError("joint hyperprior requires caller-owned limits")
    observation_metadata_cost(data, limits)
    raw = complete_topology_record(data, JointHyperpriorGraphRequest)
    declared = CausalLimits(**complete_topology_record(raw["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("joint hyperprior declared limits exceed caller limits")
    prior = topology_hyperprior_from_dict(raw["topology"])
    caps = RawDrawLimits(**complete_topology_record(raw["raw_limits"], RawDrawLimits))
    admit_topology_hyperprior(prior, declared, caps)
    task_raw = complete_topology_record(raw["task"], LinearRecipe)
    task_raw["class_offsets"] = tuple(task_raw["class_offsets"])
    task = LinearRecipe(**task_raw)
    width = raw["width"]
    if type(width) is not int or not 1 <= width <= declared.node_channels:
        raise ValueError("joint hyperprior width exceeds caller geometry")
    n = prior.minimum_nodes
    observed, target = hyperprior_readouts(task, n, width)
    shared = {
        k: raw[k]
        for k in (
            "task",
            "root_prior",
            "function_prior",
            "calibration_rows",
            "mode",
            "categorical_policy",
            "global_policy",
            "limits",
            "raw_limits",
        )
    }
    child = decode_joint_graph_request(
        {
            **shared,
            "parents": ((),) * n,
            "widths": (width,) * n,
            "observed": observed,
            "target": target,
            "noise": (raw["noise"],) * n,
            "authored_pipelines": (raw["pipeline"],) * n,
            "version": "joint-graph-request-v1",
        },
        declared,
    )
    request = JointHyperpriorGraphRequest(
        task,
        prior,
        width,
        child.calibration_rows,
        child.root_prior,
        child.function_prior,
        child.noise[0],
        child.authored_pipelines[0],
        child.mode,
        child.categorical_policy,
        child.global_policy,
        declared,
        caps,
        raw["layout"],
        raw["version"],
    )
    if encode(asdict(request)) != encode(raw):
        raise ValueError("joint hyperprior request must retain every exact nested field")
    return request


def joint_hyperprior_graph_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> JointHyperpriorGraph:
    if type(limits) is not CausalLimits:
        raise ValueError("joint hyperprior requires caller-owned limits")
    observation_metadata_cost(data, limits)
    try:
        raw = complete_topology_record(data, JointHyperpriorGraph)
        if raw["version"] != "sampled-joint-hyperprior-graph-v1":
            raise ValueError("unsupported joint hyperprior graph version")
        request = joint_hyperprior_request_from_dict(raw["request"], limits=limits)
        from r0.modules.priors.lib.topology_hyperprior_decoding import (
            sampled_topology_hyperprior_metadata,
        )

        topology = sampled_topology_hyperprior_metadata(raw["topology"], limits=request.limits)
        if (
            not construction_matches(request.topology, raw["topology"]["request"])
            or topology.limits != request.limits
            or topology.raw_limits != request.raw_limits
        ):
            raise ValueError("joint hyperprior topology binding differs")
        expected = request.for_parents(topology.parents)
        graph_raw = complete_topology_record(raw["graph"], SampledFunctionalGraph)
        if graph_raw["version"] != "sampled-functional-graph-v1" or not construction_matches(
            expected, graph_raw["request"]
        ):
            raise ValueError("joint hyperprior mechanism request binding differs")
        for key in ("initial", "final"):
            snapshots = graph_raw[key]
            if type(snapshots) not in (tuple, list) or len(snapshots) != 4:
                raise ValueError("joint mechanism graph requires four complete stream snapshots")
            for item in snapshots:
                snapshot = GeneratorSnapshot(**complete_topology_record(item, GeneratorSnapshot))
                snapshot.validate_metadata(byte_limit=request.raw_limits.state_bytes)
        # Only now can either of the five source streams be restored.
        topology.replay(limits=request.limits)
        graph = sampled_graph_from_dict(graph_raw, limits=request.limits)
        result = JointHyperpriorGraph(request, topology, graph)
        result.validate_structure()
        if not construction_matches(result, raw):
            raise ValueError("joint hyperprior graph differs from complete exact replay")
        return result
    except (
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        RecursionError,
        IndexError,
    ) as error:
        raise ValueError("malformed joint hyperprior graph archive") from error
