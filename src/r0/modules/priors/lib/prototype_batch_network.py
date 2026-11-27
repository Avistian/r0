"""Explicit batch-reference requests and identity-bound realized network populations."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.prototype_root_base import PrototypeRootBase
from r0.modules.priors.lib.prototype_root_construction import (
    PrototypeBatchReference,
    admit_prototype_plan,
    preflight_raw,
    sample_prototype_batch_reference,
)
from r0.modules.priors.lib.prototype_root_records import PrototypeRootPlan
from r0.modules.priors.lib.prototype_weight_law import PrototypeWeightLaw
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class PrototypeBatchNetworkRecipe:
    source: NetworkRecipe
    rho: float = 0.125
    weights: PrototypeWeightLaw = PrototypeWeightLaw()
    count_rule: str = "ceil-r0"
    max_prototypes: int = 256
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "prototype-batch-network-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "prototype-batch-network-request-v1"
            or type(self.source) is not NetworkRecipe
        ):
            raise ValueError("batch network requires an explicit standalone source")
        if isinstance(self.source.root, PrototypeRootBase):
            raise ValueError("batch request cannot reuse an independently frozen prototype root")
        self.admit()

    @property
    def task(self) -> LinearRecipe:
        return self.source.task

    @property
    def limits(self) -> CausalLimits:
        return self.source.limits

    @property
    def plan(self) -> PrototypeRootPlan:
        base = self.source.root
        if isinstance(base, PrototypeRootBase):
            raise ValueError("batch request cannot use an independent prototype bank")
        return PrototypeRootPlan(
            base or JointRootBase("gaussian", self.source.function.input_width),
            self.task.rows,
            self.rho,
            self.weights,
            self.count_rule,
            self.max_prototypes,
            self.raw_limits,
        )

    def admit(self) -> dict[str, int]:
        plan = self.plan
        admit_prototype_plan(plan, self.limits, batch=True)
        preflight_raw(plan, batch=True)
        graph = self.source.runtime.admit()
        parameters, storage = function_record_cost(self, self.limits)
        extra = 4 * plan.count * plan.base.width + 3 * self.task.rows * (
            plan.count + plan.base.width
        )
        work = graph["estimated_working_bytes"] + 4 * storage + 128 * extra
        if (
            graph["parameters"] + parameters + extra > self.limits.parameters
            or work > self.limits.working_bytes
        ):
            raise ValueError("batch network aggregate construction workspace limit exceeded")
        return {
            **graph,
            "parameters": graph["parameters"] + parameters + extra,
            "estimated_working_bytes": work,
        }

    def with_task(self, task: LinearRecipe) -> "PrototypeBatchNetworkRecipe":
        return replace(self, source=replace(self.source, task=task))


def batch_stream(seed: int, task_id: int, rows: int, role: str):
    if any(type(value) is not int or not 0 <= value < 2**63 for value in (seed, task_id)):
        raise ValueError("batch population identities must be exact integers in [0,2**63)")
    return functional_rng(seed, task_id, "prototype-batch-population-v1", rows, role)


@dataclass(frozen=True, slots=True)
class RealizedPrototypeBatchNetworkRecipe:
    request: PrototypeBatchNetworkRecipe
    population: PrototypeBatchReference
    seed: int
    task_id: int
    version: str = "prototype-batch-network-realized-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "prototype-batch-network-realized-v1"
            or type(self.request) is not PrototypeBatchNetworkRecipe
            or type(self.population) is not PrototypeBatchReference
        ):
            raise ValueError("invalid realized batch network record")
        self.request.admit()
        if self.population.limits != self.limits:
            raise ValueError("batch population limits differ from its request")
        if not exact_frozen_equal(self.request.plan, self.population.plan):
            raise ValueError("batch reference differs from declared row geometry or plan")
        expected = tuple(
            snapshot_generator(
                batch_stream(self.seed, self.task_id, self.task.rows, role),
                byte_limit=self.request.raw_limits.state_bytes,
            )
            for role in ("bank", "weights")
        )
        if not exact_frozen_equal(expected, self.population.initial):
            raise ValueError("batch raw population is not bound to its declared identity")
        self.population.validate_replay(limits=self.limits)
        function_record_cost(self, self.limits)

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    @property
    def limits(self) -> CausalLimits:
        return self.request.limits

    def with_task(
        self, task: LinearRecipe
    ) -> "PrototypeBatchNetworkRecipe | RealizedPrototypeBatchNetworkRecipe":
        changed = self.request.with_task(task)
        return changed if task.rows != self.task.rows else replace(self, request=changed)


def realize_prototype_batch_network(
    request: PrototypeBatchNetworkRecipe, *, seed: int, task_id: int
) -> RealizedPrototypeBatchNetworkRecipe:
    if type(request) is not PrototypeBatchNetworkRecipe:
        raise ValueError("prototype batch realization requires its explicit request")
    request.admit()
    population = sample_prototype_batch_reference(
        request.plan,
        prototype_rng=batch_stream(seed, task_id, request.task.rows, "bank"),
        weights_rng=batch_stream(seed, task_id, request.task.rows, "weights"),
        limits=request.limits,
    )
    return RealizedPrototypeBatchNetworkRecipe(request, population, seed, task_id)
