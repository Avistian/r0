"""A five-stream topology and mechanism construction, with no rejection resampling."""

from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_graph_sampling import sample_joint_graph
from r0.modules.priors.lib.joint_hyperprior_request import JointHyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_capture import (
    raw_stream_states,
    raw_transactions,
    validate_raw_streams,
)
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_topology_hyperprior import (
    SampledTopologyHyperprior,
    sample_topology_hyperprior,
)


@dataclass(frozen=True, slots=True)
class JointHyperpriorGraph:
    request: JointHyperpriorGraphRequest
    topology: SampledTopologyHyperprior
    graph: SampledFunctionalGraph
    version: str = "sampled-joint-hyperprior-graph-v1"

    @property
    def task(self):
        return self.request.task

    @property
    def limits(self):
        return self.request.limits

    @property
    def runtime(self):
        return self.graph.runtime

    def validate_structure(self) -> None:
        if (
            self.version != "sampled-joint-hyperprior-graph-v1"
            or type(self.request) is not JointHyperpriorGraphRequest
            or type(self.topology) is not SampledTopologyHyperprior
            or type(self.graph) is not SampledFunctionalGraph
        ):
            raise ValueError("joint hyperprior graph requires its complete frozen construction")
        self.request.admit()
        self.topology.validate_structure()
        self.graph.validate_structure()
        if (
            not exact_frozen_equal(self.topology.request, self.request.topology)
            or not exact_frozen_equal(self.topology.limits, self.request.limits)
            or not exact_frozen_equal(self.topology.raw_limits, self.request.raw_limits)
            or not exact_frozen_equal(
                self.graph.request, self.request.for_parents(self.topology.parents)
            )
        ):
            raise ValueError("joint hyperprior topology/mechanism request binding differs")

    def validate(self) -> None:
        self.validate_structure()
        self.topology.replay()
        self.graph.replay()

    def with_task(self, task: LinearRecipe) -> "JointHyperpriorGraph":
        request = replace(self.request, task=task)
        self.validate()
        result = replace(self, request=request, graph=self.graph.with_task(task))
        result.validate_structure()
        return result


def sample_joint_hyperprior_graph(
    request: JointHyperpriorGraphRequest,
    *,
    topology_rng: np.random.Generator,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> JointHyperpriorGraph:
    if type(request) is not JointHyperpriorGraphRequest:
        raise ValueError("joint hyperprior sampler requires a frozen request")
    request.admit()
    callers = dict(
        topology=topology_rng,
        named=named_rng,
        raw=raw_rng,
        calibration=calibration_rng,
        fit=fit_rng,
    )
    validate_raw_streams(callers)
    raw_stream_states(callers, request.raw_limits)
    for rng in callers.values():
        snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
    with raw_transactions(callers, None) as (streams, _):
        topology = sample_topology_hyperprior(
            request.topology,
            rng=streams["topology"],
            limits=request.limits,
            raw_limits=request.raw_limits,
        )
        graph = sample_joint_graph(
            request.for_parents(topology.parents),
            named_rng=streams["named"],
            raw_rng=streams["raw"],
            calibration_rng=streams["calibration"],
            fit_rng=streams["fit"],
        )
        result = JointHyperpriorGraph(request, topology, graph)
        result.validate_structure()
        return result
