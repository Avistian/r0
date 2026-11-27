"""Complete topology/mechanism records without changing explicit-DAG v1 archives."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.cauchy_graph import CauchyGraph
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.topology_graph_request import TopologyGraphRequest


@dataclass(frozen=True, slots=True)
class SampledTopologyGraph:
    request: TopologyGraphRequest
    topology: CauchyGraph
    graph: SampledFunctionalGraph
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    version: str = "sampled-topology-graph-v1"

    @property
    def runtime(self) -> FunctionalCausalRecipe:
        return self.graph.runtime

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    def validate_structure(self) -> None:
        if (
            self.version != "sampled-topology-graph-v1"
            or type(self.request) is not TopologyGraphRequest
            or type(self.topology) is not CauchyGraph
            or type(self.graph) is not SampledFunctionalGraph
        ):
            raise ValueError("sampled topology graph requires frozen versioned records")
        admit_functional_graph(
            self.runtime.nodes,
            self.request.limits,
            rows=max(self.task.rows, self.request.calibration_rows),
            metadata=(self,),
            features=self.task.features,
            outputs=len(self.request.target),
        )
        self.topology.validate()
        self.graph.validate_structure()
        if (
            self.topology.prior != self.request.topology
            or self.topology.limits != self.request.limits
            or self.graph.request
            != self.request.for_parents(
                self.topology.parents,
                limits=self.request.mechanism_limits(),
                raw_limits=self.request.mechanism_raw_limits(),
            )
        ):
            raise ValueError("sampled topology graph request binding differs")
        if (
            type(self.initial) is not tuple
            or type(self.final) is not tuple
            or len(self.initial) != 4
            or len(self.final) != 4
            or any(type(s) is not GeneratorSnapshot for s in (*self.initial, *self.final))
        ):
            raise ValueError("sampled topology graph requires four stream snapshots")
        if self.final != self.graph.final or any(
            self.initial[i] != self.graph.initial[i] for i in (0, 2, 3)
        ):
            raise ValueError("sampled topology graph stream bindings differ")
        raw = self.topology.raw
        if (
            self.initial[1].digest != raw[0].state_before
            or self.graph.initial[1].digest != raw[-1].state_after
            or self.initial[1].engine != raw[0].generator
        ):
            raise ValueError("topology draws do not precede mechanism raw construction")

    def validate(self) -> None:
        self.replay()

    def replay(self) -> "SampledTopologyGraph":
        self.validate_structure()
        result = self._reconstruct(self.request)
        if result != self:
            raise ValueError("sampled topology graph differs from complete replay")
        return result

    def with_task(self, task: LinearRecipe) -> "SampledTopologyGraph":
        self.validate()
        return self._reconstruct(replace(self.request, task=task))

    def _reconstruct(self, request: TopologyGraphRequest) -> "SampledTopologyGraph":
        from r0.modules.priors.lib.topology_graph_sampling import sample_topology_graph

        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        return sample_topology_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
