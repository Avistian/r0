"""Sample all source mechanisms on an explicit ordered DAG before numerical construction."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_sampling import sample_joint_function
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_root_plans import JointRootPlan, JointRootPrior, sample_joint_root
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import validate_path
from r0.modules.priors.lib.parent_composition_plans import (
    ParentCompositionPrior,
    ParentFunctionPlan,
    sample_parent_function,
)

_ROOT_PRIOR = JointRootPrior()
_FUNCTION_PRIOR = JointFunctionPrior()
_LIMITS = CausalLimits()


def validate_graph_geometry(
    parents: tuple[tuple[int, ...], ...],
    widths: tuple[int, ...],
    rows: int,
    limits: CausalLimits,
) -> None:
    if type(limits) is not CausalLimits:
        raise ValueError("graph planning requires frozen causal limits")
    if (
        type(parents) is not tuple
        or type(widths) is not tuple
        or not parents
        or len(parents) != len(widths)
    ):
        raise ValueError("graph requires aligned nonempty parent and width tuples")
    if type(rows) is not int or not 1 <= rows < 2**63:
        raise ValueError("graph calibration rows must be positive bounded integers")
    if len(widths) > limits.nodes:
        raise ValueError("graph planning node limit exceeded")
    if any(type(width) is not int or not 1 <= width <= limits.node_channels for width in widths):
        raise ValueError("graph planning node channel limit exceeded")
    if sum(widths) > limits.total_channels:
        raise ValueError("graph planning total channel limit exceeded")
    edges = 0
    for i, incoming in enumerate(parents):
        if type(incoming) is not tuple:
            raise ValueError("graph parents must be frozen tuples")
        edges += len(incoming)
        if (
            3 * len(widths) + edges > limits.parameters
            or 256 * (len(widths) + edges) > limits.working_bytes
        ):
            raise ValueError("graph planning metadata limit exceeded")
        if (
            type(incoming) is not tuple
            or any(type(p) is not int or not 0 <= p < i for p in incoming)
            or len(set(incoming)) != len(incoming)
        ):
            raise ValueError("graph parents must be distinct earlier nodes in declared order")
    if (
        3 * len(widths) + edges > limits.parameters
        or 256 * (len(widths) + edges) > limits.working_bytes
    ):
        raise ValueError("graph planning metadata limit exceeded")


@dataclass(frozen=True, slots=True)
class JointGraphNodePlan:
    parents: tuple[int, ...]
    width: int
    mechanism: JointRootPlan | JointFunctionPlan | ParentFunctionPlan


@dataclass(frozen=True, slots=True)
class JointGraphPlan:
    operation: tuple[str, ...]
    nodes: tuple[JointGraphNodePlan, ...]
    calibration_rows: int
    limits: CausalLimits = _LIMITS
    version: str = "joint-graph-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            self.version != "joint-graph-plan-v1"
            or type(self.nodes) is not tuple
            or any(type(node) is not JointGraphNodePlan for node in self.nodes)
        ):
            raise ValueError("graph plan requires its version and frozen node tuple")
        validate_graph_geometry(
            tuple(n.parents for n in self.nodes),
            tuple(n.width for n in self.nodes),
            self.calibration_rows,
            self.limits,
        )
        for i, node in enumerate(self.nodes):
            mechanism = node.mechanism
            if type(mechanism) not in (
                (JointFunctionPlan, ParentFunctionPlan) if node.parents else (JointRootPlan,)
            ):
                raise ValueError("graph root/function plan kind differs from ancestry")
            if mechanism.operation != (*self.operation, f"node:{i}", "mechanism"):
                raise ValueError("graph mechanism must have its owned node operation")
            if mechanism.calibration_rows != self.calibration_rows:
                raise ValueError("graph mechanism calibration rows differ")
            if isinstance(mechanism, JointRootPlan):
                valid = mechanism.width == node.width
            else:
                valid = (mechanism.input_width, mechanism.output_width) == (
                    sum(self.nodes[p].width for p in node.parents),
                    node.width,
                )
            if not valid:
                raise ValueError("graph mechanism geometry differs from ordered parent states")
        function_record_cost(self, self.limits)

    def owners(self) -> tuple[JointPlanOwner, ...]:
        result = [JointPlanOwner(self.operation, "graph", ())]
        for i, node in enumerate(self.nodes):
            result.append(JointPlanOwner((*self.operation, f"node:{i}"), "node", ("nodes", i)))
            result.extend(
                node.mechanism.owners(("nodes", i, "mechanism"))
                if isinstance(node.mechanism, ParentFunctionPlan)
                else plan_owners(node.mechanism, ("nodes", i, "mechanism"))
            )
        return tuple(result)

    def validate(
        self,
        registry: JointOperationRegistry,
        *,
        owners: tuple[JointPlanOwner, ...] | None = None,
    ) -> None:
        if type(registry) is not JointOperationRegistry:
            raise ValueError("graph plan requires its completed operation registry")
        function_record_cost((self, registry), self.limits)
        derived = self.owners()
        if owners is not None and (type(owners) is not tuple or owners != derived):
            raise ValueError("graph typed ownership references differ from retained plans")
        function_record_cost((self, registry, derived), self.limits)
        expected = tuple(owner.operation for owner in derived)
        actual = tuple(
            span.operation
            for span in registry.operations
            if span.operation[: len(self.operation)] == self.operation
        )
        if actual != expected:
            raise ValueError("graph typed ownership differs from completed operation registry")
        for node in self.nodes:
            node.mechanism.validate(registry.decisions)


def sample_joint_graph_plan(
    context: JointSamplingContext,
    name: str,
    *,
    parents: tuple[tuple[int, ...], ...],
    widths: tuple[int, ...],
    calibration_rows: int,
    root_prior: JointRootPrior = _ROOT_PRIOR,
    function_prior: JointFunctionPrior = _FUNCTION_PRIOR,
    limits: CausalLimits = _LIMITS,
    composition: ParentCompositionPrior | None = None,
) -> JointGraphPlan:
    """Sample mechanism choices; the supplied topology and widths remain explicit inputs.

    A complete builder must stage the caller context and all construction streams,
    then bind this plan to realizations and runtime before publishing caller states.
    """
    if (
        type(context) is not JointSamplingContext
        or type(root_prior) is not JointRootPrior
        or type(function_prior) is not JointFunctionPrior
    ):
        raise ValueError("graph sampling requires its shared context and frozen priors")
    validate_graph_geometry(parents, widths, calibration_rows, limits)
    with context.transaction(name) as graph:
        nodes: list[JointGraphNodePlan] = []
        for i, incoming in enumerate(parents):
            with graph.transaction(f"node:{i}") as node:
                mechanism: JointRootPlan | JointFunctionPlan | ParentFunctionPlan
                if incoming and composition is not None:
                    mechanism = sample_parent_function(
                        node,
                        "mechanism",
                        parent_widths=tuple(widths[p] for p in incoming),
                        output_width=widths[i],
                        calibration_rows=calibration_rows,
                        functions=function_prior,
                        prior=composition,
                    )
                elif incoming:
                    mechanism = sample_joint_function(
                        node,
                        "mechanism",
                        input_width=sum(widths[p] for p in incoming),
                        output_width=widths[i],
                        calibration_rows=calibration_rows,
                        prior=function_prior,
                    )
                else:
                    mechanism = sample_joint_root(
                        node,
                        "mechanism",
                        width=widths[i],
                        calibration_rows=calibration_rows,
                        prior=root_prior,
                    )
                nodes.append(JointGraphNodePlan(incoming, widths[i], mechanism))
        plan = JointGraphPlan(graph.operation, tuple(nodes), calibration_rows, limits)
        decisions = graph.freeze()
        for item in nodes:
            item.mechanism.validate(decisions)
        function_record_cost((plan, decisions), limits)
        return plan
