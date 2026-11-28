"""Independent shared-population estimates matching one frozen input mask to another."""

from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.frozen_function_evaluation import (
    admit_frozen_function,
    evaluate_admitted_function,
)
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_sampling_records import SampledNetwork
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.source_root_base import functional_root_values
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


def _mask(indices: tuple[int, ...], width: int) -> None:
    if (
        type(indices) is not tuple
        or not indices
        or any(type(i) is not int or not 0 <= i < width for i in indices)
        or indices != tuple(sorted(set(indices)))
    ):
        raise ValueError("variance matching requires sorted unique nonempty input masks")


def admit_matching(base: NetworkRecipe, rows: int) -> None:
    if type(base) is not NetworkRecipe or type(rows) is not int or not 2 <= rows <= 4096:
        raise ValueError(
            "variance matching requires an original network and 2..4096 independent rows"
        )
    limits = base.limits
    report = admit_frozen_function(base.function, rows, limits)
    parameters, storage = function_record_cost(base, limits)
    cells = rows * (base.function.input_width + 2 * base.function.output_width)
    # Retained Python matrices, replay copies, root raw records and live tensors.
    # The nested report includes hidden widths and function-specific row floors.
    extra_parameters = 8 * cells + 2 * parameters + 6 * 700
    work = report["estimated_working_bytes"] + 1024 * cells + 4 * storage + 8 * 1024**2
    if extra_parameters > limits.parameters or work > limits.working_bytes:
        raise ValueError("independent variance matching aggregate limits exceeded before draws")


def _scores(base: NetworkRecipe, population: Matrix, mask: tuple[int, ...]) -> Matrix:
    x = torch.tensor(population, dtype=torch.float64)
    keep = torch.zeros(base.function.input_width, dtype=torch.bool)
    keep[list(mask)] = True
    value = evaluate_admitted_function(base.function, torch.where(keep, x, 0.0))
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite independent matching scores; no retry")
    return frozen_matrix(value.numpy())


def _moments(scores: Matrix) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.asarray(scores, dtype=np.float64)
    if np.any(np.all(values == values[0], axis=0)):
        raise ValueError("constant independent score channel; no fallback")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean, std = values.mean(axis=0), values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("zero or nonfinite independent score dispersion; no fallback")
    return tuple(map(float, mean)), tuple(map(float, std))


@dataclass(frozen=True, slots=True)
class NetworkVarianceMatch:
    base: NetworkRecipe
    before_inputs: tuple[int, ...]
    after_inputs: tuple[int, ...]
    rows: int
    population: Matrix
    before_scores: Matrix
    after_scores: Matrix
    before_mean: tuple[float, ...]
    before_std: tuple[float, ...]
    after_mean: tuple[float, ...]
    after_std: tuple[float, ...]
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    digest: str
    version: str = "independent-network-variance-match-v1"

    def __post_init__(self) -> None:
        if self.version != "independent-network-variance-match-v1":
            raise ValueError("unknown independent network variance matching protocol")
        admit_matching(self.base, self.rows)
        _mask(self.before_inputs, self.base.function.input_width)
        _mask(self.after_inputs, self.base.function.input_width)
        if self.before_inputs == self.after_inputs:
            raise ValueError("variance matching must compare distinct masks")
        if (
            type(self.initial) is not GeneratorSnapshot
            or type(self.final) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
        ):
            raise ValueError("variance matching requires owned raw draw and snapshot custody")
        function_record_cost(self, self.base.limits)
        if matrix_shape(self.population) != (self.rows, self.base.function.input_width) or any(
            matrix_shape(values) != (self.rows, self.base.function.output_width)
            for values in (self.before_scores, self.after_scores)
        ):
            raise ValueError("matching population or score geometry differs")
        if isinstance(self.base.network, SampledNetwork):
            self.base.network.validate_replay()
        rng = self.initial.restore()
        ledger = RawDrawLedger()
        with ledger.transactions({"calibration": rng}) as (streams, capture):
            root = self.base.root or JointRootBase("gaussian", self.base.function.input_width)
            sample = functional_root_values(
                root,
                self.rows,
                streams["calibration"],
                capture=capture,
                owner=("network-variance-match",),
                limits=self.base.limits,
            )
        before = _scores(self.base, frozen_matrix(sample), self.before_inputs)
        after = _scores(self.base, frozen_matrix(sample), self.after_inputs)
        if (
            frozen_matrix(sample) != self.population
            or before != self.before_scores
            or after != self.after_scores
            or _moments(before) != (self.before_mean, self.before_std)
            or _moments(after) != (self.after_mean, self.after_std)
            or ledger.freeze() != self.raw
            or snapshot_generator(rng) != self.final
            or self.digest != record_fingerprint((self.population, before, after))
        ):
            raise ValueError("variance matching differs from independent population replay")

    def apply(self, scores: torch.Tensor) -> torch.Tensor:
        if (
            type(scores) is not torch.Tensor
            or scores.ndim != 2
            or scores.shape[0] < 1
            or scores.shape[1] != self.base.function.output_width
            or scores.dtype != torch.float64
        ):
            raise ValueError("matching requires finite float64 raw score rows")
        _, storage = function_record_cost(self, self.base.limits)
        if 4 * storage + 64 * scores.numel() > self.base.limits.working_bytes:
            raise ValueError("matching application working limit exceeded")
        if not torch.isfinite(scores).all():
            raise ValueError("matching requires finite raw scores")
        result = (scores - scores.new_tensor(self.after_mean)) / scores.new_tensor(self.after_std)
        result = result * scores.new_tensor(self.before_std) + scores.new_tensor(self.before_mean)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite matched scores; no replacement or fallback")
        return result


def fit_network_variance_match(
    base: NetworkRecipe,
    *,
    before_inputs: tuple[int, ...],
    after_inputs: tuple[int, ...],
    rows: int = 256,
    rng: np.random.Generator,
) -> NetworkVarianceMatch:
    """Fit both masks on one separately supplied population; never accepts episode data."""
    admit_matching(base, rows)
    _mask(before_inputs, base.function.input_width)
    _mask(after_inputs, base.function.input_width)
    if isinstance(base.network, SampledNetwork):
        base.network.validate_replay()
    ledger = RawDrawLedger()
    with ledger.transactions({"calibration": rng}) as (streams, capture):
        active = streams["calibration"]
        initial = snapshot_generator(active)
        sample = functional_root_values(
            base.root or JointRootBase("gaussian", base.function.input_width),
            rows,
            active,
            capture=capture,
            owner=("network-variance-match",),
            limits=base.limits,
        )
        before, after = (
            _scores(base, frozen_matrix(sample), before_inputs),
            _scores(base, frozen_matrix(sample), after_inputs),
        )
        bm, bs = _moments(before)
        am, ast = _moments(after)
        return NetworkVarianceMatch(
            base,
            before_inputs,
            after_inputs,
            rows,
            frozen_matrix(sample),
            before,
            after,
            bm,
            bs,
            am,
            ast,
            initial,
            snapshot_generator(active),
            capture.freeze(),
            record_fingerprint((frozen_matrix(sample), before, after)),
        )
