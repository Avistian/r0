"""Explicit source batch-reference emission, distinct from frozen independent fitting."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.source_root_calibration import (
    SourceRootCalibration,
    admit_root_normalization,
    root_normalization_statistics,
    validate_root_normalization_modes,
)
from r0.modules.priors.lib.source_root_laws import SourceRootLaw, source_root_from_dict
from r0.modules.priors.lib.source_root_sampling import draw_source_root

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class SourceRootBatchReference:
    law: SourceRootLaw
    modes: tuple[str, ...]
    rows: int
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    version: str = "source-root-batch-reference-v1"

    def __post_init__(self) -> None:
        if self.version != "source-root-batch-reference-v1":
            raise ValueError("invalid batch-reference root version")
        # Reuse structural validation only; no independent calibration is drawn.
        SourceRootCalibration(self.law, self.modes, self.rows, self.locations, self.scales)


def draw_source_root_batch_reference(
    law: SourceRootLaw,
    *,
    modes: tuple[str, ...],
    rows: int,
    rng: np.random.Generator,
    capture: RawDrawLedger | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
    owner: tuple[str, ...] = ("source-root-batch-reference",),
) -> tuple[NDArray[np.float64], SourceRootBatchReference]:
    """Normalize this emitted batch; changing batch composition changes its statistics.

    This is deliberately not a query-independent frozen population transform.
    The record describes this realization; it provides no apply-to-future-rows API.
    """
    validate_root_normalization_modes(law, modes, rows)
    admit_root_normalization(law, rows, limits)
    if capture is not None and type(capture) is not RawDrawLedger:
        raise ValueError("batch-reference capture must be a raw draw ledger")
    ledger = capture if capture is not None else RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        raw = draw_source_root(
            law, rows=rows, rng=stream, capture=staged, limits=limits, owner=owner
        )
        locations, scales = root_normalization_statistics(raw, modes)
        record = SourceRootBatchReference(law, modes, rows, locations, scales)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = (raw - np.asarray(locations)) / np.asarray(scales)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite batch-reference root values; no retry")
    return values, record


def source_root_batch_reference_from_dict(data: dict[str, Any]) -> SourceRootBatchReference:
    if type(data) is not dict or set(data) != {
        field.name for field in fields(SourceRootBatchReference)
    }:
        raise ValueError("batch-reference root fields differ")
    record = dict(data)
    record["law"] = source_root_from_dict(data["law"])
    for key in ("modes", "locations", "scales"):
        value = data[key]
        if type(value) not in (tuple, list) or len(value) != record["law"].width:
            raise ValueError("batch-reference root coordinate fields differ")
        record[key] = tuple(value)
    return SourceRootBatchReference(**record)
