"""Named eligible-node counts for ordered groups of observed features."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_plan_bindings import numerical_binding, operation_events
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import (
    JointSamplingRecord,
    validate_name,
    validate_path,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    name: str
    count: int

    def __post_init__(self) -> None:
        validate_name(self.name, "feature group")
        if type(self.count) is not int or not 1 <= self.count < 2**53:
            raise ValueError("feature group count must be a positive bounded integer")


def _admit(nodes: int, groups: tuple[FeatureGroup, ...], limits: CausalLimits) -> None:
    if type(limits) is not CausalLimits or type(nodes) is not int or not 1 <= nodes <= limits.nodes:
        raise ValueError("observation planning node limit exceeded")
    if (
        type(groups) is not tuple
        or not groups
        or len(groups) > limits.total_channels
        or any(type(g) is not FeatureGroup for g in groups)
        or len({g.name for g in groups}) != len(groups)
    ):
        raise ValueError("observation planning requires distinct ordered feature groups")
    count = sum(g.count for g in groups)
    if count > limits.total_channels:
        raise ValueError("observation feature channel limit exceeded")
    # Include complete permutations (not just their selected prefixes), their
    # retained copies, assignments, decision headers and bounded group names.
    size = nodes * len(groups) + count
    if 4 * size + 128 * len(groups) > limits.parameters:
        raise ValueError("observation planning parameter limit exceeded")
    if 512 * size + 8192 * len(groups) + 8 * 1024**2 > limits.working_bytes:
        raise ValueError("observation planning working memory limit exceeded")


def eligible_node_request(nodes: int) -> NumericalParameter:
    return NumericalParameter("n_feature_nodes", 1, nodes + 1, integer="floor")


@dataclass(frozen=True, slots=True)
class JointObservationPlan:
    operation: tuple[str, ...]
    nodes: int
    groups: tuple[FeatureGroup, ...]
    subsample: bool
    eligible_counts: tuple[int, ...]
    limits: CausalLimits = _LIMITS
    version: str = "joint-observation-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if len(self.operation) > 15:
            raise ValueError("observation plan must leave room for group ownership")
        _admit(self.nodes, self.groups, self.limits)
        if (
            self.version != "joint-observation-plan-v1"
            or type(self.subsample) is not bool
            or type(self.eligible_counts) is not tuple
            or len(self.eligible_counts) != len(self.groups)
            or any(type(n) is not int or not 1 <= n <= self.nodes for n in self.eligible_counts)
            or (not self.subsample and any(n != self.nodes for n in self.eligible_counts))
        ):
            raise ValueError("observation plan requires bounded eligible counts and its mode")

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if len(events) != (len(self.groups) if self.subsample else 0):
            raise ValueError("observation plan must own exactly its eligible-node decisions")
        for i, event in enumerate(events):
            if event.operation != self.operation or event.role != f"group:{i}":
                raise ValueError("eligible-node decision ownership differs")
            draw = numerical_binding(event, decisions, eligible_node_request(self.nodes))
            if draw.values != (self.eligible_counts[i],):
                raise ValueError("eligible-node count differs from its named realization")


def sample_joint_observations(
    context: JointSamplingContext,
    name: str,
    *,
    nodes: int,
    groups: tuple[FeatureGroup, ...],
    subsample: bool,
    limits: CausalLimits = _LIMITS,
) -> JointObservationPlan:
    if type(context) is not JointSamplingContext or type(subsample) is not bool:
        raise ValueError("observations require a shared context and explicit subsample mode")
    _admit(nodes, groups, limits)
    with context.transaction(name) as operation:
        counts = tuple(
            int(operation.numerical(f"group:{i}", eligible_node_request(nodes)).values[0])
            if subsample
            else nodes
            for i in range(len(groups))
        )
        plan = JointObservationPlan(operation.operation, nodes, groups, subsample, counts, limits)
        plan.validate(operation.freeze())
        return plan
