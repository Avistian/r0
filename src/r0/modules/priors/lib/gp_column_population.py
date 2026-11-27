"""Identity-bound finite GP populations and independent marginal target calibration."""

from dataclasses import asdict, dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_draw import (
    GPColumnDraw,
    GPColumnKernel,
    admit_gp_geometry,
    build_gp_columns,
    gp_matrix,
)
from r0.modules.priors.lib.gp_column_target import (
    GPColumnTarget,
    evaluate_gp_column_target,
    sample_gp_target,
)
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


@dataclass(frozen=True, slots=True)
class GPColumnRequest:
    rows: int = 32
    columns: int = 2
    dimension: int = 2
    kernels: tuple[GPColumnKernel, ...] = (GPColumnKernel(),)
    location_family: str = "uniform-r0"
    locations: Matrix = ()
    target_family: str = "linear"
    outputs: int = 1
    hidden: int = 8
    calibration_rows: int = 64
    limits: CausalLimits = CausalLimits()
    version: str = "gp-column-request-r0-v1"

    def __post_init__(self) -> None:
        if (
            any(type(v) is not int for v in (self.rows, self.columns, self.dimension))
            or not (
                1 <= self.rows <= 1024 and 1 <= self.columns <= 256 and 1 <= self.dimension <= 32
            )
            or type(self.locations) is not tuple
        ):
            raise ValueError("invalid GP population row/column/location geometry")
        if (
            self.version != "gp-column-request-r0-v1"
            or self.target_family not in ("linear", "quadratic", "neural")
            or self.location_family not in ("uniform-r0", "gaussian-r0", "grid-r0", "fixed-r0")
        ):
            raise ValueError("invalid GP population request")
        if type(self.kernels) is not tuple or any(
            type(k) is not GPColumnKernel for k in self.kernels
        ):
            raise ValueError("invalid GP population kernels")
        if any(
            type(v) is not int for v in (self.outputs, self.hidden, self.calibration_rows)
        ) or not (
            1 <= self.outputs <= 100
            and 1 <= self.hidden <= 256
            and 32 <= self.calibration_rows <= 512
        ):
            raise ValueError("invalid GP target/calibration geometry")
        if self.location_family == "fixed-r0":
            if matrix_shape(self.locations) != (self.rows, self.dimension):
                raise ValueError("fixed GP locations must match declared geometry")
        elif self.locations:
            raise ValueError("unused fixed GP locations")
        if self.location_family == "grid-r0" and self.dimension != 1:
            raise ValueError("GP grid requires one latent dimension")
        self.admit()

    def admit(self, limits: CausalLimits | None = None) -> dict[str, int]:
        extra = 8 * (
            self.columns * self.columns * self.outputs
            + (self.columns + self.outputs) * self.hidden
            + (self.rows + self.calibration_rows) * (self.columns + self.outputs + self.hidden)
            + self.outputs
        )
        report = admit_gp_geometry(
            self.rows,
            self.columns,
            self.dimension,
            len(self.kernels),
            self.limits,
            extra_numbers=extra,
        )
        if limits is not None:
            report = admit_gp_geometry(
                self.rows,
                self.columns,
                self.dimension,
                len(self.kernels),
                limits,
                extra_numbers=extra,
            )
        return report


@dataclass(frozen=True, slots=True)
class GPColumnPopulation:
    request: GPColumnRequest
    seed: int
    task_id: int
    population_id: str
    row_ids: tuple[int, ...]
    draw: GPColumnDraw
    target: GPColumnTarget
    calibration_innovations: Matrix
    calibration_inputs: Matrix
    calibration_scores: Matrix
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    raw_scores: Matrix
    scores: Matrix
    version: str = "gp-column-population-v1"


def gp_stream(seed: int, task_id: int, *role: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("gp-columns-r0-v1", seed, task_id, role)), 16)
    )


def realize_gp_columns(request: GPColumnRequest, *, seed: int, task_id: int) -> GPColumnPopulation:
    if type(request) is not GPColumnRequest or any(
        type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)
    ):
        raise ValueError("GP population requires exact request and integer identity")
    request.admit()
    n, d, c = request.rows, request.dimension, request.columns
    if request.location_family == "fixed-r0":
        locations = request.locations
    elif request.location_family == "grid-r0":
        locations = tuple((float(v),) for v in np.linspace(-1, 1, n))
    else:
        stream = gp_stream(seed, task_id, "locations", n, d, request.location_family)
        locations = gp_matrix(
            stream.uniform(-1, 1, (n, d))
            if request.location_family == "uniform-r0"
            else stream.normal(size=(n, d))
        )
    z = gp_matrix(gp_stream(seed, task_id, "column-innovations", n, c).normal(size=(n, c)))
    draw = build_gp_columns(locations, z, kernels=request.kernels, limits=request.limits)
    target = sample_gp_target(
        request.target_family,
        c,
        request.outputs,
        request.hidden,
        gp_stream(
            seed, task_id, "target", request.target_family, c, request.outputs, request.hidden
        ),
    )
    calibration_z = gp_stream(
        seed, task_id, "independent-marginal-calibration", c, request.calibration_rows
    ).normal(size=(request.calibration_rows, c))
    amplitudes = [
        request.kernels[0 if len(request.kernels) == 1 else j].amplitude for j in range(c)
    ]
    calibration = gp_matrix(calibration_z * np.asarray(amplitudes))
    calibration_scores = evaluate_gp_column_target(target, calibration, limits=request.limits)
    raw = evaluate_gp_column_target(target, draw.values, limits=request.limits)
    values = np.asarray(calibration_scores)
    magnitude = np.max(np.abs(values), axis=0)
    divisor = np.where(magnitude == 0, 1, magnitude)
    centers = (values / divisor).mean(axis=0) * divisor
    scales = (values / divisor).std(axis=0) * divisor
    # Preserve constant targets as valid draws, retaining the explicit unit fallback.
    scales = np.where(scales == 0, 1, scales)
    scores = (np.asarray(raw) - centers) / scales
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(scales).all()
    ):
        raise ValueError("nonfinite independent GP target calibration")
    population_id = record_fingerprint((asdict(request), seed, task_id))
    return GPColumnPopulation(
        request,
        seed,
        task_id,
        population_id,
        tuple(range(n)),
        draw,
        target,
        gp_matrix(calibration_z),
        calibration,
        calibration_scores,
        tuple(float(v) for v in centers),
        tuple(float(v) for v in scales),
        raw,
        gp_matrix(scores),
    )
