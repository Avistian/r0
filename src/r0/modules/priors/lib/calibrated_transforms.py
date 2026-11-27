"""Frozen source normalization transforms fitted on caller-owned independent samples."""

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.range_warp import domain_masks, kumaraswamy_warp

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class CalibratedTransform(CausalActivation):
    kind: str = "l2_normalize"
    factor: float = 1.0
    minimum: tuple[float, ...] = ()
    maximum: tuple[float, ...] = ()
    channels: int = 1
    calibration_rows: int = 0
    calibration_id: str = ""
    calibration_digest: str = ""
    calibration_version: str = "independent-transform-v1"

    def __post_init__(self) -> None:
        if (
            self.kind not in ("l2_normalize", "kumaraswamy")
            or type(self.parameters) is not tuple
            or self.calibration_version != "independent-transform-v1"
        ):
            raise ValueError("unsupported calibrated transform or parameters")
        if any(
            type(i) is not int or not 1 <= i < 2**63 for i in (self.channels, self.calibration_rows)
        ):
            raise ValueError("transform calibration requires positive channel and row counts")
        if (
            type(self.calibration_id) is not str
            or not self.calibration_id
            or type(self.calibration_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.calibration_digest) is None
        ):
            raise ValueError("transform calibration requires identity and sample SHA256")
        if not finite_number(self.factor) or not 0 < self.factor <= 1e8:
            raise ValueError("L2 normalization factor must lie in (0,1e8]")
        if self.kind == "l2_normalize":
            if self.parameters or self.minimum != () or self.maximum != ():
                raise ValueError("L2 normalization cannot contain a range or shape parameters")
        else:
            if (
                len(self.parameters) != 2
                or any(type(p) is not tuple or len(p) != 2 for p in self.parameters)
                or tuple(p[0] for p in self.parameters) != ("a", "b")
                or any(not finite_number(p[1]) or p[1] <= 0 for p in self.parameters)
                or self.factor != 1
            ):
                raise ValueError("Kumaraswamy requires positive a/b and no L2 factor")
            if any(
                type(v) is not tuple
                or len(v) != self.channels
                or any(not finite_number(x) for x in v)
                for v in (self.minimum, self.maximum)
            ) or any(low > high for low, high in zip(self.minimum, self.maximum, strict=True)):
                raise ValueError("Kumaraswamy calibration requires finite aligned ordered ranges")

    @property
    def parameter_count(self) -> int:
        return 3 + len(self.parameters) + len(self.minimum) + len(self.maximum)

    @property
    def working_extra_channels(self) -> int:
        return (16 if self.kind == "kumaraswamy" else 4) * self.channels

    def validate_width(self, width: int) -> None:
        if width != self.channels:
            raise ValueError("calibrated transform width does not match its channels")

    def _validate_input(self, value: torch.Tensor) -> None:
        if value.ndim != 2 or not value.is_floating_point():
            raise ValueError("calibrated transform requires floating rows by channels")
        self.validate_width(value.shape[1])
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite calibrated transform input; task was not retried")

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_input(value)
        result = (
            value * self.factor
            if self.kind == "l2_normalize"
            else kumaraswamy_warp(
                value, self.minimum, self.maximum, self.parameters[0][1], self.parameters[1][1]
            )
        )
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite calibrated transform output; task was not retried")
        return result

    def boundary_diagnostic(self, value: torch.Tensor) -> dict[str, object] | None:
        if self.kind != "kumaraswamy":
            return None
        self._validate_input(value)
        lower, upper = domain_masks(value, self.minimum, self.maximum)
        clamped = lower | upper
        outside = (value < value.new_tensor(self.minimum)) | (
            value > value.new_tensor(self.maximum)
        )
        flags: list[str] = []
        if torch.any(outside):
            flags.append("outside_calibration_range")
        if torch.any(clamped):
            flags.append("kumaraswamy_domain_clamp")
        return {
            "kind": self.kind,
            "rows": len(value),
            "range_epsilon": 1e-30,
            "domain_clamp_fraction": [float(v) for v in clamped.to(torch.float64).mean(dim=0)],
            "outside_calibration_fraction": [
                float(v) for v in outside.to(torch.float64).mean(dim=0)
            ],
            "constant_calibration_channels": [
                i
                for i, (low, high) in enumerate(zip(self.minimum, self.maximum, strict=True))
                if low == high
            ],
            "flags": flags,
            "meaning": "declared Kumaraswamy range clamp, not rejection or numerical repair",
        }


def _sample(
    calibration: NDArray[np.float64],
    calibration_id: str,
    limits: CausalLimits,
    *,
    parameters: int,
    working_channels_per_feature: int,
) -> tuple[torch.Tensor, str]:
    if (
        type(calibration) is not np.ndarray
        or calibration.ndim != 2
        or min(calibration.shape) < 1
        or calibration.dtype.kind != "f"
    ):
        raise ValueError("transform calibration requires a nonempty floating sample matrix")
    if type(calibration_id) is not str or not calibration_id:
        raise ValueError("transform calibration identity must be nonempty")
    if type(limits) is not CausalLimits:
        raise ValueError("transform calibration requires frozen causal limits")
    rows, width = calibration.shape
    limits.admit_dimensions(
        (width,),
        ((),),
        rows=rows,
        calibration_rows=0,
        features=width,
        outputs=width,
        extra_parameters=parameters,
        extra_working_channels=working_channels_per_feature * width,
    )
    sample = np.ascontiguousarray(calibration, dtype="<f8")
    if not np.isfinite(sample).all():
        raise ValueError("transform calibration sample must be finite")
    digest = hashlib.sha256(f"{rows},{width}:".encode() + sample.tobytes()).hexdigest()
    return torch.tensor(sample, dtype=torch.float64), digest


def calibrate_l2_normalization(
    calibration: NDArray[np.float64],
    *,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> CalibratedTransform:
    """Freeze inverse RMS row norm with epsilon 1e-8; never fit evaluation rows."""
    value, digest = _sample(
        calibration, calibration_id, limits, parameters=3, working_channels_per_feature=4
    )
    magnitude = float(value.abs().max())
    if magnitude == 0:
        factor = 1e8
    else:
        scaled_norm = float((value / magnitude).square().sum(dim=1).mean().sqrt())
        factor = (
            (1 / magnitude) / (scaled_norm + 1e-8 / magnitude)
            if magnitude >= 1
            else 1 / (magnitude * scaled_norm + 1e-8)
        )
    return CalibratedTransform(
        factor=factor,
        channels=value.shape[1],
        calibration_rows=value.shape[0],
        calibration_id=calibration_id,
        calibration_digest=digest,
    )


def calibrate_kumaraswamy(
    calibration: NDArray[np.float64],
    *,
    a: float,
    b: float,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> CalibratedTransform:
    """Freeze per-channel ranges with epsilon1e-30 and shared positive shape parameters.

    Source sampling uses log bounds0.2..5; authored shapes may be any finite positive
    values. The declared [0,1] clamp is retained and exposed by diagnostics.
    """
    if any(not finite_number(v) or v <= 0 for v in (a, b)):
        raise ValueError("Kumaraswamy shape parameters must be finite positive numbers")
    width = calibration.shape[1] if type(calibration) is np.ndarray and calibration.ndim == 2 else 0
    value, digest = _sample(
        calibration,
        calibration_id,
        limits,
        parameters=5 + 2 * width,
        working_channels_per_feature=16,
    )
    return CalibratedTransform(
        kind="kumaraswamy",
        parameters=(("a", a), ("b", b)),
        minimum=tuple(float(v) for v in value.min(dim=0).values),
        maximum=tuple(float(v) for v in value.max(dim=0).values),
        channels=value.shape[1],
        calibration_rows=value.shape[0],
        calibration_id=calibration_id,
        calibration_digest=digest,
    )
