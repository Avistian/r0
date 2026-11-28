"""Five source matrix branches bound to the shared named-decision ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_activation_plans import (
    JointActivationPlan,
    joint_activation_plan_from_dict,
    sample_joint_activation,
)
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_point_plans import (
    JointPointPlan,
    joint_point_plan_from_dict,
    sample_joint_points,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import (
    JointWeightPlan,
    joint_weight_plan_from_dict,
    sample_joint_weights,
)
from r0.modules.priors.lib.matrix_preference_prior import FAMILY_VERSIONS
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


def matrix_distance_request() -> NumericalParameter:
    return NumericalParameter("random_kernel_matrix_dist_factor", 0.1, 10.0, scale="log")


@dataclass(frozen=True, slots=True)
class JointMatrixPrior:
    families: tuple[str, ...] = tuple(FAMILY_VERSIONS)
    simple_families: tuple[str, ...] = tuple(FAMILY_VERSIONS)[:4]
    vector_family: str = "polynomial"
    scalarization: str = "scalarized-weight-batches-v1"
    version: str = "joint-matrix-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "joint-matrix-prior-v1":
            raise ValueError("unsupported joint matrix prior version")
        for choices, allowed in (
            (self.families, tuple(FAMILY_VERSIONS)),
            (self.simple_families, tuple(FAMILY_VERSIONS)[:4]),
        ):
            if (
                type(choices) is not tuple
                or not choices
                or any(type(c) is not str or c not in allowed for c in choices)
                or len(set(choices)) != len(choices)
            ):
                raise ValueError("matrix catalogs must be distinct nonempty supported tuples")
        if self.vector_family not in ("polynomial", "uniform") or self.scalarization not in (
            "scalarized-weight-batches-v1",
            "source-weight-batches-v1",
        ):
            raise ValueError("unsupported matrix vector family or batch policy")

    def request(self, *, simple: bool) -> CategoricalParameter:
        return CategoricalParameter(
            "simple_random_matrix_type" if simple else "random_matrix_type",
            tuple(FAMILY_VERSIONS[k] for k in (self.simple_families if simple else self.families)),
        )


@dataclass(frozen=True, slots=True)
class JointMatrixPlan:
    operation: tuple[str, ...]
    rows: int
    columns: int
    batch: int
    family: str
    prior: JointMatrixPrior = JointMatrixPrior()
    simple: bool = False
    weights: JointWeightPlan | None = None
    points: JointPointPlan | None = None
    distance: float | None = None
    child: JointMatrixPlan | None = None
    activation: JointActivationPlan | None = None
    version: str = "joint-matrix-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if any(
            type(n) is not int or not 1 <= n < 2**63 for n in (self.rows, self.columns, self.batch)
        ):
            raise ValueError("matrix geometry must be positive integers below 2**63")
        if (
            self.version != "joint-matrix-plan-v1"
            or type(self.prior) is not JointMatrixPrior
            or type(self.simple) is not bool
        ):
            raise ValueError("matrix plan requires its version, prior and selector scope")
        choices = self.prior.simple_families if self.simple else self.prior.families
        if self.family not in choices:
            raise ValueError("matrix family is absent from its selector catalog")
        present = tuple(
            x is not None
            for x in (self.weights, self.points, self.distance, self.child, self.activation)
        )
        required = {
            "gaussian": (False, False, False, False, False),
            "importance": (True, False, False, False, False),
            "factorized": (True, False, False, False, False),
            "kernel": (False, True, True, False, False),
            "transformed": (False, False, False, True, True),
        }
        if present != required[self.family]:
            raise ValueError("matrix plan must retain exactly its selected branch fields")
        if self.weights is not None:
            width = self.columns if self.family == "importance" else min(self.rows, self.columns)
            batch = self.batch * self.rows if self.family == "importance" else self.batch
            self._weight(self.weights, (*self.operation, "weights"), width, batch)
        if self.points is not None:
            if (
                type(self.points) is not JointPointPlan
                or self.points.operation != (*self.operation, "points")
                or self.points.count != self.batch * (self.rows + self.columns)
                or self.points.dimension != 3
                or self.points.law != "gaussian-mixture"
            ):
                raise ValueError("kernel matrix requires one shared Gaussian-mixture cloud")
            for weight in (self.points.mixture_weights, self.points.factor_weights):
                self._weight(weight, weight.operation, weight.width, weight.batch)
            if (
                self.distance is None
                or not finite_number(self.distance)
                or not 0.1 <= self.distance <= 10
            ):
                raise ValueError("matrix distance factor must lie in [0.1,10]")
        if self.child is not None:
            if (
                type(self.child) is not JointMatrixPlan
                or not self.child.simple
                or self.child.operation != (*self.operation, "child")
                or self.child.prior != self.prior
                or (self.child.rows, self.child.columns, self.child.batch)
                != (self.rows, self.columns, self.batch)
            ):
                raise ValueError("transformed matrix requires its nonrecursive simple child")
            if (
                type(self.activation) is not JointActivationPlan
                or self.activation.operation != (*self.operation, "activation")
                or self.activation.selector != "plain"
            ):
                raise ValueError("transformed matrix requires its owned plain activation")

    def _weight(
        self, weight: JointWeightPlan, path: tuple[str, ...], width: int, batch: int
    ) -> None:
        if (
            type(weight) is not JointWeightPlan
            or weight.operation != path
            or weight.width != width
            or weight.batch != batch
            or weight.family != self.prior.vector_family
            or weight.scalarization != self.prior.scalarization
        ):
            raise ValueError("matrix weight geometry, policy or ownership differs from its prior")

    @property
    def parameter_count(self) -> int:
        return (
            3
            + int(self.distance is not None)
            + sum(
                item.parameter_count
                for item in (self.weights, self.points, self.child, self.activation)
                if item is not None
            )
        )

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or events[0].operation != self.operation or events[0].role != "family":
            raise ValueError("matrix plan requires its initial family decision")
        draw = categorical_binding(events[0], decisions, self.prior.request(simple=self.simple))
        if draw.value != FAMILY_VERSIONS[self.family]:
            raise ValueError("matrix family differs from its recorded realization")
        expected = (events[0],)
        for item in (self.weights, self.points, self.child, self.activation):
            if item is not None:
                item.validate(decisions)
                expected += operation_events(decisions, item.operation)
        if self.family == "kernel":
            last = events[-1]
            if last.operation != self.operation or last.role != "distance":
                raise ValueError("kernel matrix distance must follow its complete point subtree")
            if numerical_binding(last, decisions, matrix_distance_request()).values != (
                self.distance,
            ):
                raise ValueError("matrix distance differs from its recorded realization")
            expected += (last,)
        if events != expected:
            raise ValueError("matrix plan must own exactly its ordered descendant decisions")


def sample_joint_matrix(
    context: JointSamplingContext,
    name: str,
    *,
    rows: int,
    columns: int,
    batch: int = 1,
    prior: JointMatrixPrior = JointMatrixPrior(),
    simple: bool = False,
) -> JointMatrixPlan:
    if (
        type(context) is not JointSamplingContext
        or type(prior) is not JointMatrixPrior
        or type(simple) is not bool
    ):
        raise ValueError("matrix sampling requires a shared context, prior and selector scope")
    if any(type(n) is not int or not 1 <= n < 2**63 for n in (rows, columns, batch)):
        raise ValueError("matrix geometry must be positive integers below 2**63")
    with context.transaction(name) as operation:
        selected = operation.categorical("family", prior.request(simple=simple)).value
        family = next(k for k, v in FAMILY_VERSIONS.items() if v == selected)
        weights, points, distance, child, activation = None, None, None, None, None
        if family in ("importance", "factorized"):
            weights = sample_joint_weights(
                operation,
                "weights",
                width=columns if family == "importance" else min(rows, columns),
                batch=batch * rows if family == "importance" else batch,
                family=prior.vector_family,
                scalarization=prior.scalarization,
            )
        elif family == "kernel":
            points = sample_joint_points(
                operation,
                "points",
                count=batch * (rows + columns),
                dimension=3,
                vector_family=prior.vector_family,
                scalarization=prior.scalarization,
            )
            distance = operation.numerical("distance", matrix_distance_request()).values[0]
        elif family == "transformed":
            child = sample_joint_matrix(
                operation,
                "child",
                rows=rows,
                columns=columns,
                batch=batch,
                prior=prior,
                simple=True,
            )
            activation = sample_joint_activation(operation, "activation", selector="plain")
        plan = JointMatrixPlan(
            operation.operation,
            rows,
            columns,
            batch,
            family,
            prior,
            simple,
            weights,
            points,
            distance,
            child,
            activation,
        )
        plan.validate(operation.freeze())
    return plan


def joint_matrix_plan_from_dict(
    data: dict[str, Any], *, decisions: JointSamplingRecord
) -> JointMatrixPlan:
    return _decode(data, decisions, child=False)


def _decode(
    data: dict[str, Any], decisions: JointSamplingRecord, *, child: bool
) -> JointMatrixPlan:
    fields = dict(data)
    path = fields["operation"]
    if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
        raise ValueError("matrix operation path must be a bounded array")
    fields["operation"] = tuple(path)
    prior = dict(fields.get("prior", {}))
    for name, cap in (("families", 5), ("simple_families", 4)):
        if name in prior:
            values = prior[name]
            if type(values) not in (list, tuple) or not 1 <= len(values) <= cap:
                raise ValueError("matrix prior catalogs must be bounded arrays")
            prior[name] = tuple(values)
    fields["prior"] = JointMatrixPrior(**prior)
    if fields.get("child") is not None:
        if child or fields.get("family") != "transformed":
            raise ValueError("matrix child nesting is only allowed once for transformed matrices")
        fields["child"] = _decode(fields["child"], decisions, child=True)
    for name, decoder in (
        ("weights", joint_weight_plan_from_dict),
        ("points", joint_point_plan_from_dict),
        ("activation", joint_activation_plan_from_dict),
    ):
        if fields.get(name) is not None:
            fields[name] = decoder(fields[name], decisions=decisions)
    plan = JointMatrixPlan(**fields)
    plan.validate(decisions)
    return plan
