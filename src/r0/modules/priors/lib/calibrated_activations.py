"""Frozen standardize/rescale/activate/standardize pipelines from independent samples."""

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.calibrated_transforms import CalibratedTransform, calibration_sample
from r0.modules.priors.lib.causal_activations import CausalActivation, primitive_activation
from r0.modules.priors.lib.causal_limits import CausalLimits

_EPSILON = 1e-4
_DEFAULT_LIMITS = CausalLimits()
STANDARDIZATION_CONVENTIONS = {
    "independent-standardization-v1": (0, _EPSILON),
    "independent-tabicl-standardization-v1": (1, 1e-6),
}


def _finite(value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite calibrated activation stage; task was not retried")
    return value


def _center_scale(
    value: torch.Tensor, mean: tuple[float, ...], inverse: tuple[float, ...]
) -> torch.Tensor:
    center = value.new_tensor(mean)
    factor = value.new_tensor(inverse)
    difference = value - center
    result = difference * factor
    # Avoid an intermediate overflow when the normalized result is representable.
    overflow = ~torch.isfinite(difference)
    if torch.any(overflow):
        alternative = ((value / 2 - center / 2) * factor) * 2
        result = torch.where(overflow, alternative, result)
    return _finite(result)


@dataclass(frozen=True, slots=True)
class CalibratedActivation(CausalActivation):
    input_mean: tuple[float, ...] = ()
    input_inverse_std: tuple[float, ...] = ()
    shift: tuple[float, ...] = ()
    scale: float = 1.0
    output_mean: tuple[float, ...] = ()
    output_inverse_std: tuple[float, ...] = ()
    shift_rows: tuple[int, ...] = ()
    calibration_rows: int = 0
    calibration_id: str = ""
    calibration_digest: str = ""
    calibration_version: str = "independent-standard-activation-v1"

    def __post_init__(self) -> None:
        CausalActivation.__post_init__(self)
        if self.calibration_version != "independent-standard-activation-v1":
            raise ValueError("unsupported activation calibration version")
        width = len(self.input_mean)
        if width < 1 or any(
            type(values) is not tuple
            or len(values) != width
            or any(not finite_number(v) for v in values)
            for values in (
                self.input_mean,
                self.input_inverse_std,
                self.shift,
                self.output_mean,
                self.output_inverse_std,
            )
        ):
            raise ValueError("calibrated activation channel constants must be finite and aligned")
        if any(
            v <= 0 or v > 1 / _EPSILON
            for v in (
                *self.input_inverse_std,
                *self.output_inverse_std,
            )
        ):
            raise ValueError("calibrated activation inverse scales must be in (0,10000]")
        if not finite_number(self.scale) or not 1 <= self.scale <= 10:
            raise ValueError("source activation rescale must lie in [1,10]")
        if (
            type(self.calibration_rows) is not int
            or not 1 <= self.calibration_rows < 2**63
            or type(self.shift_rows) is not tuple
            or len(self.shift_rows) != width
            or any(
                type(i) is not int or not 0 <= i < self.calibration_rows for i in self.shift_rows
            )
        ):
            raise ValueError("calibration shift rows must identify one fitted row per channel")
        if (
            type(self.calibration_id) is not str
            or not self.calibration_id
            or type(self.calibration_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.calibration_digest) is None
        ):
            raise ValueError("calibration requires sample identity and SHA-256 digest")

    @property
    def parameter_count(self) -> int:
        # Five channel vectors, sampled row indices, scalar scale and row count.
        return len(self.parameters) + 6 * len(self.input_mean) + 2

    @property
    def working_extra_channels(self) -> int:
        return 10 * len(self.input_mean)

    def validate_width(self, width: int) -> None:
        if width != len(self.input_mean):
            raise ValueError("calibrated activation width does not match its channels")

    def diagnostic_input(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("calibrated activation requires rows by channels")
        self.validate_width(value.shape[1])
        _finite(value)
        standardized = _center_scale(value, self.input_mean, self.input_inverse_std)
        return _finite(self.scale * (standardized + value.new_tensor(self.shift)))

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        transformed = primitive_activation(self.diagnostic_input(value), self)
        return _center_scale(transformed, self.output_mean, self.output_inverse_std)


def _statistics(
    value: torch.Tensor, *, correction: int = 0, epsilon: float = _EPSILON
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    _finite(value)
    # Subtract a represented sample value before scaling so a large common
    # offset cannot quantize away a small but representable channel spread.
    offset = value - value[0]
    safe = torch.isfinite(offset).all(dim=0)
    reference = torch.where(safe, value[0], torch.zeros_like(value[0]))
    # Opposite-sign extremes can overflow subtraction; their spread is large,
    # so magnitude scaling of the original values is safe from cancellation.
    offset = torch.where(safe, offset, value)
    magnitude = offset.abs().max(dim=0).values
    divisor = torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)
    scaled = offset / divisor
    mean = _finite(reference + scaled.mean(dim=0) * divisor)
    std = _finite(scaled.std(dim=0, correction=correction) * divisor)
    inverse = _finite(1 / (std + epsilon))
    return tuple(float(v) for v in mean), tuple(float(v) for v in inverse)


@dataclass(frozen=True, slots=True)
class CalibratedStandardization(CausalActivation):
    """Frozen channel statistics with versioned population or TabICL conventions."""

    kind: str = "standardize"
    mean: tuple[float, ...] = ()
    inverse_std: tuple[float, ...] = ()
    calibration_rows: int = 0
    calibration_id: str = ""
    calibration_digest: str = ""
    calibration_version: str = "independent-standardization-v1"

    def __post_init__(self) -> None:
        if self.kind != "standardize" or self.parameters != ():
            raise ValueError("standalone standardization has no primitive shape parameters")
        if (
            type(self.calibration_version) is not str
            or self.calibration_version not in STANDARDIZATION_CONVENTIONS
        ):
            raise ValueError("unsupported standalone standardization version")
        correction, epsilon = STANDARDIZATION_CONVENTIONS[self.calibration_version]
        if not self.mean or any(
            type(values) is not tuple
            or len(values) != len(self.mean)
            or any(not finite_number(v) for v in values)
            for values in (self.mean, self.inverse_std)
        ):
            raise ValueError("standardization constants must be finite aligned channel tuples")
        if any(not 0 < value <= 1 / epsilon for value in self.inverse_std):
            raise ValueError("standardization inverse scales exceed the version's epsilon bound")
        if (
            type(self.calibration_rows) is not int
            or not 1 + correction <= self.calibration_rows < 2**63
        ):
            raise ValueError("standardization requires enough calibration rows for its convention")
        if (
            type(self.calibration_id) is not str
            or not self.calibration_id
            or type(self.calibration_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.calibration_digest) is None
        ):
            raise ValueError("standardization requires sample identity and SHA-256 digest")

    @property
    def parameter_count(self) -> int:
        return 2 * len(self.mean) + 1

    @property
    def working_extra_channels(self) -> int:
        return 10 * len(self.mean)

    def validate_width(self, width: int) -> None:
        if width != len(self.mean):
            raise ValueError("standardization width does not match its channels")

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or not value.is_floating_point():
            raise ValueError("standardization requires floating rows by channels")
        self.validate_width(value.shape[1])
        return _center_scale(_finite(value), self.mean, self.inverse_std)


def calibrate_standardization(
    calibration: NDArray[np.float64],
    *,
    calibration_id: str,
    profile: str = "population",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> CalibratedStandardization:
    """Fit a caller-owned independent sample once; evaluation never refits constants."""
    if profile not in ("population", "tabicl"):
        raise ValueError("standardization profile must be population or tabicl")
    version = (
        "independent-tabicl-standardization-v1"
        if profile == "tabicl"
        else "independent-standardization-v1"
    )
    correction, epsilon = STANDARDIZATION_CONVENTIONS[version]
    if (
        type(calibration) is np.ndarray
        and calibration.ndim == 2
        and calibration.shape[0] <= correction
    ):
        raise ValueError("TabICL sample standardization requires at least two calibration rows")
    width = calibration.shape[1] if type(calibration) is np.ndarray and calibration.ndim == 2 else 0
    value, digest = calibration_sample(
        calibration,
        calibration_id,
        limits,
        parameters=2 * width + 1,
        working_channels_per_feature=10,
    )
    mean, inverse = _statistics(value, correction=correction, epsilon=epsilon)
    return CalibratedStandardization(
        mean=mean,
        inverse_std=inverse,
        calibration_rows=value.shape[0],
        calibration_id=calibration_id,
        calibration_digest=digest,
        calibration_version=version,
    )


def calibrate_source_activation(
    activation: CausalActivation,
    calibration: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    scale: float,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> CalibratedActivation:
    """Fit only the supplied independent sample; evaluation never refits constants.

    Callers own the sample's independent stream identity. This function has no task,
    context, query or target input. The digest covers shape and little-endian float64
    sample bytes; the sample itself is not duplicated in the fitted mechanism.
    """
    if type(activation) is not CausalActivation or type(rng) is not np.random.Generator:
        raise ValueError("calibration requires a primitive activation and injected Generator")
    if type(limits) is not CausalLimits:
        raise ValueError("activation calibration requires frozen causal limits")
    return _fit_source_activation(
        activation,
        calibration,
        rng=rng,
        shift_rows=None,
        scale=scale,
        calibration_id=calibration_id,
        limits=limits,
    )


def reconstruct_source_activation(
    activation: CausalActivation,
    calibration: NDArray[np.float64],
    *,
    shift_rows: tuple[int, ...],
    scale: float,
    calibration_id: str,
) -> CalibratedActivation:
    """Reconstruct an already admitted fit from retained row choices, without RNG.

    Enclosing construction/decoding must admit the complete record and sample
    before calling this helper, including all intermediate fitting buffers.
    """
    if type(activation) is not CausalActivation or type(shift_rows) is not tuple:
        raise ValueError("activation reconstruction requires a primitive and retained row tuple")
    return _fit_source_activation(
        activation,
        calibration,
        rng=None,
        shift_rows=shift_rows,
        scale=scale,
        calibration_id=calibration_id,
        limits=None,
    )


def _fit_source_activation(
    activation: CausalActivation,
    calibration: NDArray[np.float64],
    *,
    rng: np.random.Generator | None,
    shift_rows: tuple[int, ...] | None,
    scale: float,
    calibration_id: str,
    limits: CausalLimits | None,
) -> CalibratedActivation:
    if (
        type(calibration) is not np.ndarray
        or calibration.ndim != 2
        or min(calibration.shape) < 1
        or calibration.dtype.kind != "f"
    ):
        raise ValueError("activation calibration requires a nonempty floating sample matrix")
    rows, width = calibration.shape
    if type(calibration_id) is not str or not calibration_id:
        raise ValueError("calibration sample identity must be nonempty")
    if not finite_number(scale) or not 1 <= scale <= 10:
        raise ValueError("source activation rescale must lie in [1,10]")
    if shift_rows is not None and (
        len(shift_rows) != width or any(type(i) is not int or not 0 <= i < rows for i in shift_rows)
    ):
        raise ValueError("activation shift rows must match calibration geometry")
    if limits is not None:
        limits.admit_dimensions(
            (width,),
            ((),),
            rows=rows,
            calibration_rows=0,
            features=width,
            outputs=width,
            extra_parameters=activation.parameter_count + 6 * width + 2,
            extra_working_channels=10 * width,
        )
    sample = np.ascontiguousarray(calibration, dtype="<f8")
    if not np.isfinite(sample).all():
        raise ValueError("activation calibration sample must be finite")
    digest = hashlib.sha256(f"{rows},{width}:".encode() + sample.tobytes()).hexdigest()
    value = torch.tensor(sample, dtype=torch.float64)
    input_mean, input_inverse = _statistics(value)
    standardized = _center_scale(value, input_mean, input_inverse)
    indices = (
        tuple(int(i) for i in rng.integers(rows, size=width)) if rng is not None else shift_rows
    )
    assert indices is not None
    shift = tuple(-float(standardized[i, j]) for j, i in enumerate(indices))
    argument = _finite(scale * (standardized + standardized.new_tensor(shift)))
    result = primitive_activation(argument, activation)
    output_mean, output_inverse = _statistics(result)
    return CalibratedActivation(
        activation.kind,
        activation.parameters,
        input_mean,
        input_inverse,
        shift,
        scale,
        output_mean,
        output_inverse,
        indices,
        rows,
        calibration_id,
        digest,
    )


def activation_from_dict(data: dict[str, Any]) -> CausalActivation:
    fields = dict(data)
    fields["parameters"] = tuple(tuple(p) for p in fields.get("parameters", ()))
    if fields.get("calibration_version") in (
        "independent-standardization-v1",
        "independent-tabicl-standardization-v1",
    ):
        for key in ("mean", "inverse_std"):
            fields[key] = tuple(fields.get(key, ()))
        return CalibratedStandardization(**fields)
    if fields.get("calibration_version") == "independent-transform-v1":
        for key in ("minimum", "maximum"):
            fields[key] = tuple(fields.get(key, ()))
        return CalibratedTransform(**fields)
    if "calibration_version" not in fields:
        return CausalActivation(**fields)
    if fields["calibration_version"] != "independent-standard-activation-v1":
        raise ValueError("unsupported activation calibration version")
    for key in (
        "input_mean",
        "input_inverse_std",
        "shift",
        "output_mean",
        "output_inverse_std",
        "shift_rows",
    ):
        fields[key] = tuple(fields[key])
    return CalibratedActivation(**fields)
