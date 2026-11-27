"""Explicit surgical changes retain their original graph and intervention location."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction, function_geometry
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.native_causal import NativeCausalNode, NativeCausalRecipe
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph
from r0.modules.priors.lib.sampled_parent_graph import SampledParentGraph
from r0.modules.priors.lib.sampled_topology_graph import SampledTopologyGraph

type CausalBase = (
    FunctionalCausalRecipe
    | NativeCausalRecipe
    | SampledFunctionalGraph
    | SampledGeometryGraph
    | SampledNativeGraph
    | SampledParentGraph
    | SampledTopologyGraph
)


@dataclass(frozen=True, slots=True)
class HardIntervention:
    node: int
    channels: tuple[int, ...] = ()
    values: tuple[float, ...] = ()
    category: str | None = None
    location: str = "state"
    version: str = "hard-causal-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "hard-causal-intervention-v1"
            or type(self.node) is not int
            or self.node < 0
            or self.location not in ("state", "readout", "state_and_readout")
            or type(self.channels) is not tuple
            or any(type(c) is not int or c < 0 for c in self.channels)
            or len(set(self.channels)) != len(self.channels)
            or type(self.values) is not tuple
            or any(not finite_number(v) for v in self.values)
            or self.category is not None
            and type(self.category) is not str
            or (
                self.category is None
                and (not self.values or len(self.values) != len(self.channels))
            )
            or (
                self.category is not None
                and (self.channels or self.values or self.location != "state")
            )
        ):
            raise ValueError(
                "hard intervention requires explicit channels/values or one native category"
            )


@dataclass(frozen=True, slots=True)
class SoftIntervention:
    node: int
    function: FrozenFunction | None = None
    noise_scale: tuple[float, ...] | None = None
    noise_response: NoiseResponse | None = None
    exogenous: ExogenousDistribution | None = None
    version: str = "soft-causal-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "soft-causal-intervention-v1"
            or type(self.node) is not int
            or self.node < 0
        ):
            raise ValueError("soft intervention requires its version and node")
        if all(
            v is None
            for v in (self.function, self.noise_scale, self.noise_response, self.exogenous)
        ):
            raise ValueError("soft intervention must declare an equation or innovation change")
        if self.function is not None:
            function_geometry(self.function)


def causal_runtime(base: object) -> FunctionalCausalRecipe | NativeCausalRecipe:
    if isinstance(base, (FunctionalCausalRecipe, NativeCausalRecipe)):
        return base
    if isinstance(base, SampledNativeGraph):
        return base.recipe
    if isinstance(
        base,
        (SampledFunctionalGraph, SampledGeometryGraph, SampledParentGraph, SampledTopologyGraph),
    ):
        return base.runtime
    raise ValueError("interventions require a frozen functional or native causal graph")


@dataclass(frozen=True, slots=True)
class CausalInterventionRecipe:
    base: CausalBase
    interventions: tuple[HardIntervention | SoftIntervention, ...]
    mode: str = "matched"
    population: int = 1
    version: str = "causal-intervention-v1"

    @property
    def task(self):
        return self.base.task

    def __post_init__(self) -> None:
        if (
            self.version != "causal-intervention-v1"
            or self.mode not in ("matched", "population")
            or type(self.population) is not int
            or not 1 <= self.population < 2**63
            or type(self.interventions) is not tuple
            or not self.interventions
        ):
            raise ValueError("intervention recipe requires explicit changes and comparison mode")
        runtime = causal_runtime(self.base)
        occupied: set[tuple[int, str, int]] = set()
        soft: set[int] = set()
        hard: set[int] = set()
        for change in self.interventions:
            if type(change) not in (HardIntervention, SoftIntervention) or change.node >= len(
                runtime.nodes
            ):
                raise ValueError("intervention refers to an invalid node")
            node = runtime.nodes[change.node]
            native = node if isinstance(node, NativeCausalNode) else None
            equation = node.equation if isinstance(node, NativeCausalNode) else node
            if isinstance(change, SoftIntervention):
                if change.node in soft or change.node in hard:
                    raise ValueError("overlapping hard/soft or duplicate soft interventions")
                soft.add(change.node)
                kwargs = {
                    name: getattr(change, name)
                    for name in ("function", "noise_scale", "noise_response", "exogenous")
                    if getattr(change, name) is not None
                }
                revised = replace(equation, **kwargs)
                if (revised.function.input_width, revised.function.output_width) != (
                    equation.function.input_width,
                    equation.function.output_width,
                ):
                    raise ValueError("soft replacement must preserve local equation geometry")
            else:
                if change.node in soft:
                    raise ValueError("overlapping hard and soft interventions")
                hard.add(change.node)
                if native and native.domain.kind != "continuous":
                    if change.category not in native.domain.vocabulary:
                        raise ValueError(
                            "native category intervention must name a declared vocabulary value"
                        )
                    keys = [(change.node, "category", 0)]
                else:
                    if change.category is not None or any(
                        c >= equation.function.output_width for c in change.channels
                    ):
                        raise ValueError("continuous intervention requires valid channels")
                    if native and change.location != "state":
                        raise ValueError(
                            "native continuous variables have one emitted state location"
                        )
                    locations = (
                        ("state", "readout")
                        if change.location == "state_and_readout"
                        else (change.location,)
                    )
                    keys = [(change.node, loc, c) for loc in locations for c in change.channels]
                if any(key in occupied for key in keys):
                    raise ValueError("overlapping hard intervention locations")
                occupied.update(keys)
        # Validate replacement equations jointly, preserving topology and calibration.
        self.admit()

    def admit(self) -> dict[str, int]:
        original = causal_runtime(self.base)
        effective = self.effective_runtime()
        metadata, storage = function_record_cost((self.base, self.interventions), original.limits)
        channels = sum(
            (
                n.equation.function.input_width
                + 8 * n.equation.function.output_width
                + 4 * n.state_width
            )
            if isinstance(n, NativeCausalNode)
            else (n.function.input_width + 10 * n.function.output_width)
            for n in original.nodes
        )
        working = (
            original.admit()["estimated_working_bytes"]
            + effective.admit()["estimated_working_bytes"]
            + 4 * storage
            + 32 * original.task.rows * channels
        )
        if working > original.limits.working_bytes:
            raise ValueError("causal intervention comparison working memory limit exceeded")
        return {"parameters": metadata, "estimated_working_bytes": working}

    def effective_runtime(self) -> FunctionalCausalRecipe | NativeCausalRecipe:
        original = causal_runtime(self.base)
        replacements = {c.node: c for c in self.interventions if isinstance(c, SoftIntervention)}

        def equation(index: int, node: FunctionalCausalNode) -> FunctionalCausalNode:
            change = replacements.get(index)
            if change is None:
                return node
            kwargs = {
                name: getattr(change, name)
                for name in ("function", "noise_scale", "noise_response", "exogenous")
                if getattr(change, name) is not None
            }
            return replace(node, **kwargs)

        if isinstance(original, NativeCausalRecipe):
            return replace(
                original,
                nodes=tuple(
                    replace(n, equation=equation(i, n.equation))
                    for i, n in enumerate(original.nodes)
                ),
            )
        return FunctionalCausalRecipe(
            original.task,
            tuple(equation(i, n) for i, n in enumerate(original.nodes)),
            original.observed,
            original.target,
            original.limits,
        )
