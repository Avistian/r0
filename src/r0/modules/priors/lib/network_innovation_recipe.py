"""Versioned stochastic standalone network with source-to-runtime custody."""

from dataclasses import dataclass

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.network_innovation_compile import compile_network_innovations
from r0.modules.priors.lib.network_innovation_plan import NetworkInnovationPlan
from r0.modules.priors.lib.network_innovation_sampling_records import SampledNetworkInnovations
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_sampling_records import SampledNetwork


@dataclass(frozen=True, slots=True)
class NetworkInnovationRecipe(FunctionalCausalRecipe):
    source: NetworkRecipe | None = None
    plan: NetworkInnovationPlan | None = None
    sampling: SampledNetworkInnovations | None = None
    version: str = "standalone-network-innovations-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "standalone-network-innovations-v1"
            or self.source is None
            or self.plan is None
        ):
            raise ValueError("stochastic network requires retained source and innovation plan")
        expected, mapping = compile_network_innovations(self.source, self.plan)
        if (
            self.task != self.source.task
            or self.limits != self.source.limits
            or self.nodes != expected
            or self.observed != tuple((0, j) for j in range(self.source.function.input_width))
            or self.target
            != tuple((mapping[-1], j) for j in range(self.source.function.output_width))
        ):
            raise ValueError("stochastic network runtime differs from retained source and plan")
        FunctionalCausalRecipe(self.task, self.nodes, self.observed, self.target, self.limits)
        self.admit()
        if self.sampling is not None:
            if (
                type(self.sampling) is not SampledNetworkInnovations
                or self.sampling.plan != self.plan
            ):
                raise ValueError("network statistics and plan disagree")
            self.sampling.validate_replay(self.source)
        if isinstance(self.source.network, SampledNetwork):
            self.source.network.validate_replay()

    @property
    def block_nodes(self) -> tuple[int, ...]:
        assert self.source is not None
        assert self.plan is not None
        return compile_network_innovations(self.source, self.plan)[1]

    def admit(self, *, evaluation_rows: int | None = None) -> dict[str, int]:
        if evaluation_rows is not None and (
            type(evaluation_rows) is not int or not 1 <= evaluation_rows < 2**63
        ):
            raise ValueError("network evaluation rows must be bounded positive integers")
        return admit_functional_graph(
            self.nodes,
            self.limits,
            rows=max(self.task.rows, evaluation_rows or 0),
            metadata=(self,),
            features=self.task.features,
            outputs=len(self.target),
        )


def with_network_innovations(
    source: NetworkRecipe,
    plan: NetworkInnovationPlan,
    *,
    sampling: SampledNetworkInnovations | None = None,
) -> NetworkInnovationRecipe:
    nodes, mapping = compile_network_innovations(source, plan)
    return NetworkInnovationRecipe(
        source.task,
        nodes,
        tuple((0, j) for j in range(source.function.input_width)),
        tuple((mapping[-1], j) for j in range(source.function.output_width)),
        source.limits,
        source=source,
        plan=plan,
        sampling=sampling,
    )
