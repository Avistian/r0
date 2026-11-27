"""Explicit mechanism sharing and readout layout for sampled-size forests."""

from dataclasses import asdict, dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.hyperprior_graph_admission import (
    admit_hyperprior_graph,
    hyperprior_readouts,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.topology_hyperprior import (
    FixedSizeTopologyHyperprior,
    TopologyHyperprior,
    TopologyHyperpriorRequest,
)


@dataclass(frozen=True, slots=True)
class HyperpriorGraphRequest:
    task: LinearRecipe
    topology: TopologyHyperpriorRequest
    root_template: FunctionalCausalNode
    edge_template: FunctionalCausalNode
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    layout: str = "prefix-features-suffix-targets-r0-v1"
    mechanism_sharing: str = "shared-root-and-edge-templates-r0-v1"
    version: str = "hyperprior-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            type(self.task) is not LinearRecipe
            or type(self.topology) not in (TopologyHyperprior, FixedSizeTopologyHyperprior)
            or type(self.root_template) is not FunctionalCausalNode
            or type(self.edge_template) is not FunctionalCausalNode
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or self.version != "hyperprior-graph-request-v1"
            or self.layout != "prefix-features-suffix-targets-r0-v1"
            or self.mechanism_sharing != "shared-root-and-edge-templates-r0-v1"
        ):
            raise ValueError("hyperprior graph requires complete frozen templates and profile")
        width = self.root_template.function.output_width
        if (
            self.root_template.parents != ()
            or self.edge_template.parents != (0,)
            or self.edge_template.function.input_width != width
            or self.edge_template.function.output_width != width
        ):
            raise ValueError("template geometry requires one equal-width predecessor per edge")
        self.admit()

    def admit(self, *, retained: object = None) -> dict[str, int]:
        from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
        from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior

        admit_topology_hyperprior(self.topology, self.limits, self.raw_limits)
        # Bound complete typed metadata before copying it to raw decoder previews.
        observation_metadata_cost((self, retained), self.limits)
        return admit_hyperprior_graph(
            self.task,
            self.topology,
            asdict(self.root_template),
            asdict(self.edge_template),
            self.root_template.function.output_width,
            self.limits,
            self.raw_limits,
            retained=retained,
        )

    def runtime(self, parents: tuple[tuple[int, ...], ...]) -> FunctionalCausalRecipe:
        if not self.topology.minimum_nodes <= len(parents) <= self.topology.maximum_nodes:
            raise ValueError("realized topology size is outside the retained prior")
        observed, target = hyperprior_readouts(
            self.task, len(parents), self.root_template.function.output_width
        )
        nodes = tuple(
            replace(self.edge_template if incoming else self.root_template, parents=incoming)
            for incoming in parents
        )
        return FunctionalCausalRecipe(self.task, nodes, observed, target, self.limits)
