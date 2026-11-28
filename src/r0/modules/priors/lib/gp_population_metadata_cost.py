"""Symbolic finite GP receipt sizes under the existing frozen metadata accounting law."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_draw import GPColumnDraw
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation, GPColumnRequest
from r0.modules.priors.lib.gp_column_target import GPColumnTarget


@dataclass(frozen=True, slots=True)
class MetadataCost:
    numbers: int
    storage: int
    nodes: int
    depth: int

    def admit(self, limits: CausalLimits) -> None:
        if (
            self.numbers > limits.parameters
            or self.storage > limits.working_bytes
            or self.nodes > 16 * limits.parameters + 64 * limits.nodes
            or self.depth > 32
        ):
            raise ValueError("GP retained metadata aggregate exceeds limits")


def sequence_cost(item: MetadataCost, count: int) -> MetadataCost:
    return MetadataCost(
        count * item.numbers,
        32 + count * item.storage,
        1 + count * item.nodes,
        1 + item.depth if count else 0,
    )


def record_cost(kind: type, values: dict[str, MetadataCost]) -> MetadataCost:
    names = tuple(field.name for field in fields(kind))
    if set(names) != set(values):
        raise ValueError("symbolic GP metadata schema differs from retained record")
    return MetadataCost(
        sum(v.numbers for v in values.values()),
        32 + sum(len(name.encode()) for name in names) + sum(v.storage for v in values.values()),
        1 + len(names) + sum(v.nodes for v in values.values()),
        1 + max((v.depth for v in values.values()), default=0),
    )


def retained_cost(value: Any, limits: CausalLimits) -> MetadataCost:
    """Count real metadata without copying arrays, using the same scalar/container charges."""
    visited = 0

    def visit(item: Any, depth: int) -> MetadataCost:
        nonlocal visited
        visited += 1
        if depth > 32 or visited > 16 * limits.parameters + 64 * limits.nodes:
            raise ValueError("GP retained metadata traversal exceeds limits")
        if type(item) in (int, float, bool):
            cost = MetadataCost(1, 64, 1, 0)
        elif type(item) is str:
            cost = MetadataCost(0, len(item.encode("utf-8")), 1, 0)
        elif item is None:
            cost = MetadataCost(0, 0, 1, 0)
        elif type(item) in (tuple, list):
            children = [visit(child, depth + 1) for child in cast(tuple[Any, ...], item)]
            cost = MetadataCost(
                sum(c.numbers for c in children),
                32 + sum(c.storage for c in children),
                1 + sum(c.nodes for c in children),
                1 + max((c.depth for c in children), default=-1),
            )
        elif type(item) is dict or is_dataclass(item) and not isinstance(item, type):
            pairs = (
                cast(dict[str, Any], item).items()
                if type(cast(Any, item)) is dict
                else (
                    (field.name, getattr(cast(Any, item), field.name))
                    for field in fields(cast(Any, item))
                )
            )
            children: list[MetadataCost] = []
            for key, child in pairs:
                if type(key) is not str:
                    raise ValueError("GP metadata keys must be strings")
                children.extend((visit(key, depth + 1), visit(child, depth + 1)))
            cost = MetadataCost(
                sum(c.numbers for c in children),
                32 + sum(c.storage for c in children),
                1 + sum(c.nodes for c in children),
                1 + max((c.depth for c in children), default=-1),
            )
        else:
            raise ValueError("GP metadata requires retained records and primitive values")
        cost.admit(limits)
        return cost

    return visit(value, 0)


def population_metadata_cost(request: GPColumnRequest) -> MetadataCost:
    request.admit()
    limits = request.limits
    number = MetadataCost(1, 64, 1, 0)
    empty = sequence_cost(number, 0)

    def matrix(rows: int, columns: int) -> MetadataCost:
        return sequence_cost(sequence_cost(number, columns), rows)

    def text(value: str) -> MetadataCost:
        return retained_cost(value, limits)

    n, c, d, o, h, q = (
        request.rows,
        request.columns,
        request.dimension,
        request.outputs,
        request.hidden,
        request.calibration_rows,
    )
    draw = record_cost(
        GPColumnDraw,
        dict(
            locations=matrix(n, d),
            innovations=matrix(n, c),
            kernels=retained_cost(request.kernels, limits),
            covariance=sequence_cost(matrix(n, n), len(request.kernels)),
            factors=sequence_cost(matrix(n, n), len(request.kernels)),
            values=matrix(n, c),
            version=text("gp-column-draw-v1"),
        ),
    )
    neural, quadratic = request.target_family == "neural", request.target_family == "quadratic"
    target = record_cost(
        GPColumnTarget,
        dict(
            family=text(request.target_family),
            weights=matrix(h if neural else c, o),
            bias=sequence_cost(number, o),
            quadratic=sequence_cost(matrix(c, c), o) if quadratic else empty,
            hidden=matrix(c, h) if neural else empty,
            hidden_bias=sequence_cost(number, h) if neural else empty,
            version=text("gp-column-target-r0-v1"),
        ),
    )
    return record_cost(
        GPColumnPopulation,
        dict(
            request=retained_cost(request, limits),
            seed=number,
            task_id=number,
            population_id=text("0" * 64),
            row_ids=sequence_cost(number, n),
            draw=draw,
            target=target,
            calibration_innovations=matrix(q, c),
            calibration_inputs=matrix(q, c),
            calibration_scores=matrix(q, o),
            centers=sequence_cost(number, o),
            scales=sequence_cost(number, o),
            raw_scores=matrix(n, o),
            scores=matrix(n, o),
            version=text("gp-column-population-v1"),
        ),
    )


def gp_column_population_metadata_cost(request: GPColumnRequest) -> dict[str, int]:
    """Exact future receipt cost without sampling, covariance work or dummy arrays.

    Numeric leaves cost64 bytes and containers32 bytes, matching existing frozen
    metadata admission. The result includes field names, fixed64-character hashes
    and every nested request/draw/target/calibration matrix. It is an admission
    accounting report, not a measured allocator/RSS or JSON byte size.
    """
    if type(request) is not GPColumnRequest:
        raise ValueError("future GP metadata requires an exact population request")
    cost = population_metadata_cost(request)
    cost.admit(request.limits)
    return dict(numbers=cost.numbers, storage=cost.storage, nodes=cost.nodes, depth=cost.depth)


def future_gp_root_cost(
    request: GPColumnRequest, base: Any, bindings: Any, limits: CausalLimits
) -> MetadataCost:
    from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe

    return record_cost(
        GPRootDAGRecipe,
        dict(
            source=population_metadata_cost(request),
            base=retained_cost(base, limits),
            bindings=retained_cost(bindings, limits),
            limits=retained_cost(limits, limits),
            version=retained_cost("gp-root-dag-v1", limits),
        ),
    )


def gp_root_dag_metadata_cost(
    request: GPColumnRequest, base: Any, bindings: Any, *, limits: CausalLimits
) -> dict[str, int]:
    """Cost the whole future GP-root envelope without realizing its population.

    Callers separately validate graph/binding semantics and admit runtime geometry;
    this report includes complete source, actual base, bindings, limits and keys.
    """
    if type(request) is not GPColumnRequest or type(limits) is not CausalLimits:
        raise ValueError("future GP root metadata requires exact request and caps")
    request.admit(limits)
    cost = future_gp_root_cost(request, base, bindings, limits)
    cost.admit(limits)
    return dict(numbers=cost.numbers, storage=cost.storage, nodes=cost.nodes, depth=cost.depth)
