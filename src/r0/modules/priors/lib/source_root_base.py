"""Versioned independent root laws for functional graph inputs."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_root_base import JointRootBase, draw_root_base
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.source_root_calibration import (
    SourceRootCalibration,
    source_root_calibration_from_dict,
)
from r0.modules.priors.lib.source_root_laws import SourceRootLaw, source_root_from_dict
from r0.modules.priors.lib.source_root_sampling import draw_source_root

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class SourceRootBase:
    law: SourceRootLaw
    calibration: SourceRootCalibration | None = None
    version: str = "source-root-base-v1"

    def __post_init__(self) -> None:
        if self.version != "source-root-base-v1" or type(self.law) is not SourceRootLaw:
            raise ValueError("invalid source root base version or law")
        if self.calibration is not None and (
            type(self.calibration) is not SourceRootCalibration or self.calibration.law != self.law
        ):
            raise ValueError("source root calibration must match the frozen law")

    @property
    def width(self) -> int:
        return self.law.width

    @property
    def moment_order(self) -> float:
        return self.law.moment_order


def source_root_base_from_dict(data: dict[str, Any]) -> SourceRootBase:
    if type(data) is not dict or set(data) != {field.name for field in fields(SourceRootBase)}:
        raise ValueError("source root base fields differ")
    return SourceRootBase(
        source_root_from_dict(data["law"]),
        source_root_calibration_from_dict(data["calibration"])
        if data["calibration"] is not None
        else None,
        data["version"],
    )


def functional_root_values(
    base: JointRootBase | SourceRootBase,
    rows: int,
    rng: np.random.Generator,
    *,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = ("functional-root",),
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> NDArray[np.float64]:
    if type(base) is JointRootBase:
        return np.asarray(draw_root_base(base, rows, rng, capture=capture, owner=owner).values)
    if type(base) is not SourceRootBase:
        raise ValueError("unsupported functional root descriptor")
    ledger = capture if capture is not None else RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        raw = draw_source_root(
            base.law, rows=rows, rng=stream, capture=staged, owner=owner, limits=limits
        )
        values = base.calibration.apply(raw, limits=limits) if base.calibration else raw
    return values
