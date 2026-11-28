"""Conditional integer quota allocation with floors at every hierarchy level."""

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class CoverageHierarchy:
    dimensions: tuple[str, ...]
    weights: tuple[tuple[tuple[str, ...], int], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.dimensions) is not tuple
            or not 1 <= len(self.dimensions) <= 4
            or any(type(d) is not str or not 1 <= len(d) <= 96 for d in self.dimensions)
            or len(set(self.dimensions)) != len(self.dimensions)
            or type(self.weights) is not tuple
            or len(self.weights) > 16384
        ):
            raise ValueError("hierarchy needs 1..4 distinct dimensions and bounded branch weights")
        paths: set[tuple[str, ...]] = set()
        for item in self.weights:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("hierarchy weights require path/weight pairs")
            path, weight = item
            if (
                type(path) is not tuple
                or not 1 <= len(path) <= len(self.dimensions)
                or any(type(v) is not str or not 1 <= len(v) <= 96 for v in path)
                or path in paths
                or type(weight) is not int
                or not 0 <= weight <= 10**12
            ):
                raise ValueError("invalid or duplicate hierarchy branch weight")
            paths.add(path)


def hierarchy_from_dict(value: Any) -> CoverageHierarchy:
    if type(value) is not dict:
        raise ValueError("hierarchy must be an object")
    if set(cast(dict[str, Any], value)) != {"dimensions", "weights"}:
        raise ValueError("hierarchy fields differ")
    data = cast(dict[str, Any], value)
    if type(data["dimensions"]) is not list or type(data["weights"]) is not list:
        raise ValueError("hierarchy dimensions and weights must be arrays")
    weights: list[tuple[tuple[str, ...], int]] = []
    for raw_item in cast(list[Any], data["weights"]):
        if type(raw_item) is not list:
            raise ValueError("hierarchy weight must be an array")
        item = cast(list[Any], raw_item)
        if len(item) != 2 or type(item[0]) is not list:
            raise ValueError("hierarchy weight must be a path array and integer")
        weights.append((tuple(cast(list[Any], item[0])), item[1]))
    return CoverageHierarchy(tuple(cast(list[Any], data["dimensions"])), tuple(weights))


def quota_counts(keys: list[str], minimums: list[int], weights: list[int], total: int) -> list[int]:
    residual = total - sum(minimums)
    denominator = sum(weights)
    if residual < 0 or (residual and not denominator):
        raise ValueError("coverage minimums exceed total or residual allocation has no weight")
    divisions = [divmod(residual * w, denominator) if denominator else (0, 0) for w in weights]
    counts = [minimum + d[0] for minimum, d in zip(minimums, divisions, strict=True)]
    for index in sorted(range(len(keys)), key=lambda i: (-divisions[i][1], keys[i]))[
        : total - sum(counts)
    ]:
        counts[index] += 1
    return counts


def hierarchical_counts(
    hierarchy: CoverageHierarchy,
    keys: list[str],
    tags: list[dict[str, str]],
    minimums: list[int],
    weights: list[int],
    total: int,
) -> list[int]:
    if any(not set(hierarchy.dimensions) <= set(tag) for tag in tags):
        raise ValueError("hierarchy dimension is not declared by every stratum")
    values = [tuple(tag[d] for d in hierarchy.dimensions) for tag in tags]
    valid_paths = {value[:depth] for value in values for depth in range(1, len(value) + 1)}
    branch_weights = dict(hierarchy.weights)
    if not set(branch_weights) <= valid_paths:
        raise ValueError("hierarchy weight names a nonexistent branch")
    counts = [0] * len(keys)

    def allocate(indices: list[int], prefix: tuple[str, ...], budget: int) -> None:
        depth = len(prefix)
        if depth == len(hierarchy.dimensions):
            assigned = quota_counts(
                [keys[i] for i in indices],
                [minimums[i] for i in indices],
                [weights[i] for i in indices],
                budget,
            )
            for i, count in zip(indices, assigned, strict=True):
                counts[i] = count
            return
        children: dict[str, list[int]] = {}
        for i in indices:
            children.setdefault(values[i][depth], []).append(i)
        names = sorted(children)
        assigned = quota_counts(
            names,
            [sum(minimums[i] for i in children[name]) for name in names],
            [branch_weights.get((*prefix, name), 1) for name in names],
            budget,
        )
        for name, count in zip(names, assigned, strict=True):
            allocate(children[name], (*prefix, name), count)

    allocate(list(range(len(keys))), (), total)
    return counts
