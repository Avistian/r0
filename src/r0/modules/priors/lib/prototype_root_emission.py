"""Shared row weights map a frozen prototype bank to continuous convex mixtures."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.prototype_root_records import FrozenPrototypeRoot, admit_prototype_work
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class PrototypeRootDraw:
    values: Matrix
    weights: Matrix
    counts: tuple[tuple[int, ...], ...] | None
    version: str = "prototype-root-draw-v1"

    def __post_init__(self) -> None:
        rows, _ = matrix_shape(self.values)
        weight_rows, count = matrix_shape(self.weights)
        if (
            self.version != "prototype-root-draw-v1"
            or rows != weight_rows
            or any(v < 0 for row in self.weights for v in row)
            or any(
                not math.isclose(math.fsum(row), 1.0, rel_tol=1e-12, abs_tol=1e-12)
                for row in self.weights
            )
        ):
            raise ValueError("prototype weights must be nonnegative convex row vectors")
        if self.counts is not None:
            if (
                type(self.counts) is not tuple
                or len(self.counts) != rows
                or any(
                    type(row) is not tuple
                    or len(row) != count
                    or any(type(v) is not int or not 0 <= v < 2**31 for v in row)
                    for row in self.counts
                )
            ):
                raise ValueError("prototype counts must be frozen nonnegative integer rows")
            trials = sum(self.counts[0])
            if (
                not 1 <= trials < 2**31
                or any(sum(row) != trials for row in self.counts)
                or self.weights != tuple(tuple(v / trials for v in row) for row in self.counts)
            ):
                raise ValueError("prototype count sums or normalized weights differ")


def draw_prototype_root(
    root: FrozenPrototypeRoot,
    *,
    rows: int,
    rng: np.random.Generator,
    capture: RawDrawLedger | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
    owner: tuple[str, ...] = ("prototype-emission",),
) -> PrototypeRootDraw:
    if type(root) is not FrozenPrototypeRoot:
        raise ValueError("prototype emission requires its frozen bank")
    admit_prototype_work(root.count, root.width, rows, limits)
    if capture is not None and type(capture) is not RawDrawLedger:
        raise ValueError("prototype emission capture must be a raw ledger")
    ledger = capture if capture is not None else RawDrawLedger()
    spec = root.weights.draw_spec(root.count, rows, owner)
    ledger.admit_draws(rng, (spec,))
    with ledger.transaction(rng) as (stream, staged):
        raw = staged.draw(
            stream,
            owner=spec.owner,
            role=spec.role,
            distribution=spec.distribution,
            shape=spec.shape,
            parameters=spec.parameters,
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            if root.weights.mode == "dirichlet-concentration-r0":
                total = raw.sum(axis=1, keepdims=True)
                if np.any(raw <= 0) or np.any(total <= 0) or not np.isfinite(total).all():
                    raise ValueError("degenerate prototype Gamma weights; no retry")
                weights = raw / total
                counts = None
            else:
                weights = raw.astype(np.float64) / root.weights.effective_trials()
                counts = tuple(tuple(int(v) for v in row) for row in raw)
            values = weights @ np.asarray(root.prototypes, dtype=np.float64)
        if not np.isfinite(weights).all() or not np.isfinite(values).all():
            raise ValueError("nonfinite prototype convex mixture; no retry")
        return PrototypeRootDraw(frozen_matrix(values), frozen_matrix(weights), counts)
