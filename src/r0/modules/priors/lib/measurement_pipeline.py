"""Ordered bounded r0 measurement maps, distinct from latent structural functions."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import raw_transactions

_DEFAULT = CausalLimits()
_DEFAULTS: dict[str, dict[str, float]] = {
    "affine": {"scale": 1.0, "shift": 0.0},
    "linear-mix": {},
    "signed-power": {"power": 1.0},
    "signed-log1p": {},
    "tanh": {},
    "sin": {},
    "exp": {},
    "clip": {"lower": -3.0, "upper": 3.0},
    "round": {"step": 1.0},
    "gaussian-noise": {"scale": 1.0},
    "shared-gaussian-noise": {"scale": 1.0},
    "heteroscedastic-noise": {"scale": 1.0, "slope": 1.0},
    "outliers": {"probability": 0.01, "scale": 10.0},
    "student-noise": {"df": 3.0, "scale": 1.0},
}


def _mix_parameter(name: str, width: int) -> bool:
    if len(name) > 12:
        return False
    parts = name.split("_")
    if not ((parts[0] == "w" and len(parts) == 3) or (parts[0] == "bias" and len(parts) == 2)):
        return False
    return all(
        value.isascii() and value.isdigit() and str(int(value)) == value and int(value) < width
        for value in parts[1:]
    )


@dataclass(frozen=True, slots=True)
class MeasurementStage:
    kind: str
    columns: tuple[int, ...]
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or self.kind not in _DEFAULTS
            or type(self.columns) is not tuple
            or not 1 <= len(self.columns) <= 256
            or any(type(c) is not int or not 0 <= c < 256 for c in self.columns)
            or len(set(self.columns)) != len(self.columns)
            or type(self.parameters) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or not finite_number(pair[1])
                for pair in self.parameters
            )
        ):
            raise ValueError("invalid bounded measurement stage")
        supplied = dict(self.parameters)
        unknown = (
            any(not _mix_parameter(name, len(self.columns)) for name in supplied)
            if self.kind == "linear-mix"
            else bool(supplied.keys() - _DEFAULTS[self.kind].keys())
        )
        if len(supplied) != len(self.parameters) or unknown:
            raise ValueError("duplicate or unknown measurement parameters")
        p = self.controls
        if (
            self.kind == "clip"
            and p["lower"] > p["upper"]
            or self.kind in ("round", "signed-power")
            and next(iter(p.values())) <= 0
            or self.kind.endswith("noise")
            and p["scale"] < 0
            or self.kind == "heteroscedastic-noise"
            and p["slope"] < 0
            or self.kind == "outliers"
            and (not 0 <= p["probability"] <= 1 or p["scale"] < 0)
            or self.kind == "student-noise"
            and p["df"] <= 0
        ):
            raise ValueError("measurement parameter lies outside its support")

    @property
    def controls(self) -> dict[str, float]:
        return {**_DEFAULTS[self.kind], **dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class MeasurementDraw:
    stages: tuple[MeasurementStage, ...]
    values: NDArray[np.float64]
    binding: str
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    version: str = "r0-measurement-draw-v1"


def _apply(
    stage: MeasurementStage, x: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    p = stage.controls
    match stage.kind:
        case "affine":
            return p["scale"] * x + p["shift"]
        case "linear-mix":
            width = len(stage.columns)
            matrix = np.eye(width, dtype=np.float64)
            bias = np.zeros(width, dtype=np.float64)
            for name, value in stage.parameters:
                parts = name.split("_")
                if parts[0] == "w":
                    matrix[int(parts[1]), int(parts[2])] = value
                else:
                    bias[int(parts[1])] = value
            return x @ matrix.T + bias
        case "signed-power":
            return np.sign(x) * np.abs(x) ** p["power"]
        case "signed-log1p":
            return np.sign(x) * np.log1p(np.abs(x))
        case "tanh":
            return np.tanh(x)
        case "sin":
            return np.sin(x)
        case "exp":
            return np.exp(x)
        case "clip":
            return np.clip(x, p["lower"], p["upper"])
        case "round":
            return np.rint(x / p["step"]) * p["step"]
        case "gaussian-noise":
            return x + p["scale"] * rng.normal(size=x.shape)
        case "shared-gaussian-noise":
            return x + p["scale"] * rng.normal(size=(len(x), 1))
        case "heteroscedastic-noise":
            return x + (p["scale"] + p["slope"] * np.abs(x)) * rng.normal(size=x.shape)
        case "outliers":
            active = rng.random(x.shape) < p["probability"]
            noise = rng.normal(size=x.shape)
            return x + active * p["scale"] * noise
        case "student-noise":
            return x + p["scale"] * rng.standard_t(p["df"], size=x.shape)
        case _:
            raise ValueError("unsupported measurement stage")


def draw_measurements(
    stages: tuple[MeasurementStage, ...],
    inputs: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT,
) -> MeasurementDraw:
    """Transform in declared order; fail without retry or clipping nonfinite results.

    Only listed clipping stages clip. Every noise stage consumes its declared
    draws even at zero scale or probability. No rows are used to fit a transform.
    """
    if (
        type(stages) is not tuple
        or not 1 <= len(stages) <= 64
        or any(type(s) is not MeasurementStage for s in stages)
        or type(limits) is not CausalLimits
        or type(inputs) is not np.ndarray
        or inputs.dtype != np.float64
        or inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 20000
        or not 1 <= inputs.shape[1] <= 256
    ):
        raise ValueError("measurement pipeline requires bounded stages and float64 matrix")
    if any(max(stage.columns) >= inputs.shape[1] for stage in stages):
        raise ValueError("measurement column lies outside input schema")
    matrix_sizes = [len(s.columns) ** 2 + len(s.columns) for s in stages if s.kind == "linear-mix"]
    if sum(matrix_sizes) > limits.parameters:
        raise ValueError("measurement matrix parameters exceed limits")
    if 65536 + inputs.size * 128 + max(matrix_sizes, default=0) * 16 > limits.working_bytes:
        raise ValueError("measurement workspace exceeds limits")
    if not np.isfinite(inputs).all():
        raise ValueError("measurement inputs must be finite before masking")
    digest = hashlib.sha256(json.dumps([asdict(s) for s in stages], sort_keys=True).encode())
    digest.update(str(inputs.shape).encode())
    digest.update(np.ascontiguousarray(inputs, dtype="<f8").tobytes())
    with raw_transactions({"measurement": rng}, None) as (streams, _):
        active = streams["measurement"]
        initial = snapshot_generator(active)
        values = inputs.copy()
        for stage in stages:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                output = _apply(stage, values[:, stage.columns], active)
            if not np.isfinite(output).all():
                raise ValueError(f"measurement stage {stage.kind} produced nonfinite values")
            values[:, stage.columns] = output
        return MeasurementDraw(
            stages, values, digest.hexdigest(), initial, snapshot_generator(active)
        )
