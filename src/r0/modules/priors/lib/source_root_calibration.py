"""Independent frozen normalization; source batch-reference semantics stay separate."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.source_root_laws import SourceRootLaw, source_root_from_dict
from r0.modules.priors.lib.source_root_sampling import draw_source_root

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class SourceRootCalibration:
    law: SourceRootLaw
    modes: tuple[str, ...]
    rows: int
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    version: str = "source-root-independent-calibration-v1"

    def __post_init__(self) -> None:
        if (
            type(self.law) is not SourceRootLaw
            or self.version != "source-root-independent-calibration-v1"
        ):
            raise ValueError("invalid independent root calibration law or version")
        _validate_modes(self.law, self.modes, self.rows)
        if (
            type(self.locations) is not tuple
            or type(self.scales) is not tuple
            or len(self.locations) != self.law.width
            or len(self.scales) != self.law.width
            or any(not finite_number(v) for v in self.locations)
            or any(not finite_number(v) or v <= 0 for v in self.scales)
        ):
            raise ValueError("root calibration requires finite locations and positive scales")
        for mode, location, scale in zip(self.modes, self.locations, self.scales, strict=True):
            if (mode == "identity" and location != 0) or (mode != "standardize" and scale != 1):
                raise ValueError("unused root normalization controls")

    def apply(
        self, values: NDArray[np.float64], *, limits: CausalLimits = _DEFAULT_LIMITS
    ) -> NDArray[np.float64]:
        if type(values) is not np.ndarray or values.ndim != 2 or values.shape[1] != self.law.width:
            raise ValueError("root normalization requires a row by coordinate array")
        _admit(self.law, values.shape[0], limits)
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("root normalization requires finite binary64 input")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (values - np.asarray(self.locations)) / np.asarray(self.scales)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite normalized root values; no retry")
        return result


def _validate_modes(law: SourceRootLaw, modes: tuple[str, ...], rows: int) -> None:
    if type(law) is not SourceRootLaw or type(modes) is not tuple or len(modes) != law.width:
        raise ValueError("root normalization requires one frozen mode per coordinate")
    if any(
        type(mode) is not str or mode not in ("identity", "center", "standardize") for mode in modes
    ):
        raise ValueError("unknown root normalization mode")
    if type(rows) is not int or not 1 <= rows < 2**63 or ("standardize" in modes and rows < 2):
        raise ValueError("root calibration rows must admit sample standard deviation")


def _admit(law: SourceRootLaw, rows: int, limits: CausalLimits) -> None:
    if type(limits) is not CausalLimits or type(rows) is not int or not 1 <= rows < 2**63:
        raise ValueError("invalid root calibration limits or rows")
    limits.admit_dimensions(
        (law.width,),
        ((),),
        rows=rows,
        calibration_rows=rows,
        features=law.width,
        outputs=law.width,
        extra_parameters=12 * law.width
        + rows * law.width
        + sum(len(x.probabilities) for x in law.coordinates),
        extra_working_channels=8 * law.width,
    )


def calibrate_source_root(
    law: SourceRootLaw,
    *,
    modes: tuple[str, ...],
    rows: int,
    rng: np.random.Generator,
    capture: RawDrawLedger | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
    owner: tuple[str, ...] = ("source-root-calibration",),
) -> SourceRootCalibration:
    """Draw independent calibration rows once; retain raw draws in caller capture.

    Standardization uses sample standard deviation (ddof=1). Exact constants,
    underflowed or nonfinite scales reject without epsilon, retries or RNG advance.
    This is an explicit r0 lifecycle, not the source implementation's batch refit.
    """
    _validate_modes(law, modes, rows)
    _admit(law, rows, limits)
    if capture is not None and type(capture) is not RawDrawLedger:
        raise ValueError("root calibration capture must be a raw draw ledger")
    ledger = capture if capture is not None else RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        values = draw_source_root(
            law, rows=rows, rng=stream, capture=staged, limits=limits, owner=owner
        )
        locations: list[float] = []
        scales: list[float] = []
        for column, mode in enumerate(modes):
            channel = values[:, column]
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                location = 0.0 if mode == "identity" else float(channel.mean())
                scale = float(channel.std(ddof=1)) if mode == "standardize" else 1.0
            if mode == "standardize" and (
                np.all(channel == channel[0]) or not np.isfinite(scale) or scale <= 0
            ):
                raise ValueError("degenerate root calibration; no retry")
            locations.append(location)
            scales.append(scale)
        result = SourceRootCalibration(law, modes, rows, tuple(locations), tuple(scales))
        # Reject a calibration whose own transformed values overflow as well.
        result.apply(values, limits=limits)
    return result


def source_root_calibration_from_dict(data: dict[str, Any]) -> SourceRootCalibration:
    """Decode authored frozen transforms; raw-custody envelopes validate provenance separately."""
    if type(data) is not dict or set(data) != {
        field.name for field in fields(SourceRootCalibration)
    }:
        raise ValueError("root calibration fields differ")
    record = dict(data)
    record["law"] = source_root_from_dict(data["law"])
    for key in ("modes", "locations", "scales"):
        value = data[key]
        if type(value) not in (tuple, list) or len(value) != record["law"].width:
            raise ValueError("root calibration coordinate fields differ")
        record[key] = tuple(value)
    return SourceRootCalibration(**record)
