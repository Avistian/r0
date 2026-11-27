"""Explicit per-node population replacements retain a valid original DAG as counterfactual."""

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.native_causal import NativeCausalNode, VariableDomain
from r0.modules.priors.lib.native_vector_records import NativeVectorNode
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_forest_plan import (
    RecursiveForestPlan,
    recursive_effective_limits,
)
from r0.modules.priors.lib.recursive_population_ancestors import ANCESTOR_TYPES, PopulationAncestors
from r0.modules.priors.lib.recursive_task_preflight import admit_recursive_tasks


@dataclass(frozen=True, slots=True)
class RecursiveNodePopulationPlan:
    node: int
    forest: RecursiveForestPlan
    version: str = "recursive-node-population-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-node-population-plan-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.forest) is not RecursiveForestPlan
            or self.forest.lifecycle != "fresh-forest-per-row"
        ):
            raise ValueError("node population plan requires an explicit row-owned forest")


def population_equations(base: PopulationAncestors) -> tuple[FunctionalCausalNode, ...]:
    return tuple(n if isinstance(n, FunctionalCausalNode) else n.equation for n in base.nodes)


def validate_population_emission(
    base: PopulationAncestors, binding: RecursiveNodePopulationPlan
) -> None:
    """A classifier's winner indicator is decoded directly, never treated as random logits."""
    if binding.forest.task != "classification":
        return
    node = base.nodes[binding.node]
    domain = (
        node.domain
        if isinstance(node, NativeCausalNode)
        else node.domains[0]
        if isinstance(node, NativeVectorNode) and len(node.domains) == 1
        else None
    )
    equation = population_equations(base)[binding.node]
    if (
        type(domain) is not VariableDomain
        or domain.kind not in ("binary", "categorical")
        or domain.emission != "argmax"
        or len(domain.vocabulary) != binding.forest.classes
        or equation.pipeline is not None
        or any(equation.noise_scale)
    ):
        raise ValueError(
            "population classifier requires one unmodified deterministic native label emission"
        )


@dataclass(frozen=True, slots=True)
class RecursivePopulationGraphRequest:
    base: PopulationAncestors
    replacements: tuple[RecursiveNodePopulationPlan, ...]
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "recursive-population-graph-request-v1"

    @property
    def task(self):
        return self.base.task

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-graph-request-v1"
            or type(self.base) not in ANCESTOR_TYPES
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or getattr(self.base, "target_calibration_rows", 0) != 0
            or type(self.replacements) is not tuple
            or not self.replacements
            or any(type(p) is not RecursiveNodePopulationPlan for p in self.replacements)
            or tuple(p.node for p in self.replacements)
            != tuple(sorted({p.node for p in self.replacements}))
        ):
            raise ValueError("population graph requires ordered unique explicit node replacements")
        equations = population_equations(self.base)
        for binding in self.replacements:
            if binding.node >= len(equations):
                raise ValueError("population replacement node is outside the original DAG")
            equation = equations[binding.node]
            output = binding.forest.classes or binding.forest.output_width
            if (
                binding.forest.populations != self.task.rows
                or binding.forest.input_width != equation.function.input_width
                or output != equation.function.output_width
            ):
                raise ValueError(
                    "population replacement must preserve node geometry and original rows"
                )
            validate_population_emission(self.base, binding)
        self.admit()

    def admit(self, *, limits: CausalLimits | None = None) -> dict[str, int]:
        effective = recursive_effective_limits(self.limits, limits or self.limits)
        effective = recursive_effective_limits(effective, self.base.limits)
        admit_recursive_tasks(
            [asdict(self.base)],
            storage=0,
            numbers=0,
            replay_work=0,
            limits=effective,
            retain_node_stages=True,
        )
        base = self.base.admit()
        reports = [p.forest.admit(effective, self.raw_limits) for p in self.replacements]
        raw_draws = sum(r["draws"] for r in reports)
        raw_numbers = sum(r["numbers"] for r in reports)
        metadata, storage = function_record_cost(self, effective)
        channels = sum(
            p.forest.input_width + (p.forest.classes or p.forest.output_width)
            for p in self.replacements
        )
        channels += sum(
            n.function.input_width + n.function.output_width
            for n in population_equations(self.base)
        )
        retained = self.task.rows * channels
        parameters = (
            base["parameters"] + sum(r["parameters"] for r in reports) + metadata + 8 * retained
        )
        working = (
            base["estimated_working_bytes"]
            + sum(r["estimated_working_bytes"] for r in reports)
            + 8 * storage
            + 2048 * retained
        )
        admit_recursive_tasks(
            [asdict(self.base)],
            storage=storage,
            numbers=metadata + sum(r["parameters"] for r in reports) + 8 * retained,
            replay_work=sum(r["estimated_working_bytes"] for r in reports) + 2048 * retained,
            limits=effective,
            retain_node_stages=True,
        )
        if (
            len(self.base.nodes) + sum(r["nodes"] for r in reports) > effective.nodes
            or parameters > effective.parameters
            or working > effective.working_bytes
            or raw_draws > self.raw_limits.draws
            or raw_numbers > self.raw_limits.numbers
        ):
            raise ValueError(
                "aggregate population DAG construction, replay or retained rows exceed limits"
            )
        return {
            "parameters": parameters,
            "estimated_working_bytes": working,
            "raw_draws": raw_draws,
            "raw_numbers": raw_numbers,
        }
