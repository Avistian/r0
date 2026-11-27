"""Fixed target normalization from a separately keyed graph population."""

import hashlib
import re
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, cast

import numpy as np
import torch

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.calibration import stable_calibration


@dataclass(frozen=True, slots=True)
class FunctionalTargetCalibration:
    rows: int
    center: tuple[float, ...]
    scale: tuple[float, ...]
    constant_channels: tuple[int, ...]
    sample_digest: str
    prefix_fingerprint: str
    stream: str = "target-calibration-v1"
    version: str = "functional-target-calibration-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "functional-target-calibration-v1"
            or self.stream != "target-calibration-v1"
        ):
            raise ValueError("unsupported functional target calibration law")
        if type(self.rows) is not int or not 32 <= self.rows <= 65536:
            raise ValueError("target calibration requires 32..65536 independent rows")
        if (
            type(self.center) is not tuple
            or not self.center
            or type(self.scale) is not tuple
            or len(self.center) != len(self.scale)
            or any(not finite_number(v) for v in self.center)
            or any(not finite_number(v) or v <= 0 for v in self.scale)
        ):
            raise ValueError(
                "target calibration requires aligned finite centers and positive scales"
            )
        if (
            type(self.constant_channels) is not tuple
            or any(
                type(j) is not int or not 0 <= j < len(self.center) for j in self.constant_channels
            )
            or tuple(sorted(set(self.constant_channels))) != self.constant_channels
            or any(self.center[j] != 0 or self.scale[j] != 1 for j in self.constant_channels)
        ):
            raise ValueError("constant calibration channels must retain identity scaling")
        if any(
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (self.sample_digest, self.prefix_fingerprint)
        ):
            raise ValueError("target calibration requires sample and frozen-prefix fingerprints")

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != len(self.center):
            raise ValueError("target calibration width differs from the declared readout")
        center, scale = value.new_tensor(self.center), value.new_tensor(self.scale)
        difference = value - center
        result = difference / scale
        overflow = ~torch.isfinite(difference)
        if torch.any(overflow):
            alternative = (value / 2 - center / 2) / (scale / 2)
            result = torch.where(overflow, alternative, result)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite calibrated target; task was not retried")
        return result


def fit_target_calibration(
    raw: torch.Tensor, prefix_fingerprint: str
) -> FunctionalTargetCalibration:
    """The graph caller supplies an admitted independent CPU population, never task rows."""
    values = np.ascontiguousarray(raw.numpy(), dtype="<f8")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nonfinite target calibration population; task was not retried")
    rows, width = values.shape
    center, scale = np.zeros(width), np.ones(width)
    constant: list[int] = []
    for channel in range(width):
        column = values[:, channel : channel + 1]
        if np.all(column == column[0]):
            constant.append(channel)
        else:
            mean, std = stable_calibration(column)
            center[channel], scale[channel] = mean[0], std[0]
    digest = hashlib.sha256(f"{rows},{width}:".encode())
    digest.update(memoryview(values).cast("B"))
    return FunctionalTargetCalibration(
        rows,
        tuple(map(float, center)),
        tuple(map(float, scale)),
        tuple(constant),
        digest.hexdigest(),
        prefix_fingerprint,
    )


def target_calibration_from_dict(data: Any, width: int) -> FunctionalTargetCalibration:
    if type(data) is not dict:
        raise ValueError("target calibration must retain exactly its versioned record fields")
    raw = cast(dict[str, Any], data)
    if set(raw) != {f.name for f in dataclass_fields(FunctionalTargetCalibration)}:
        raise ValueError("target calibration must retain exactly its versioned record fields")
    fields = dict(raw)
    for name in ("center", "scale", "constant_channels"):
        if type(fields[name]) not in (list, tuple):
            raise ValueError("target calibration vectors must be arrays")
        if len(fields[name]) > width or name != "constant_channels" and len(fields[name]) != width:
            raise ValueError("target calibration vectors must match the declared target width")
        fields[name] = tuple(fields[name])
    return FunctionalTargetCalibration(**fields)
