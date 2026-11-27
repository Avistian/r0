"""Atomic finite root emissions with raw integers retained before transformations."""

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.source_root_laws import SourceRootLaw

_DEFAULT_LIMITS = CausalLimits()


def draw_source_root(
    law: SourceRootLaw,
    *,
    rows: int,
    rng: np.random.Generator,
    capture: RawDrawLedger | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
    owner: tuple[str, ...] = ("source-root",),
) -> NDArray[np.float64]:
    if type(law) is not SourceRootLaw or type(limits) is not CausalLimits:
        raise ValueError("root emission requires frozen law and limits")
    if type(rows) is not int or not 1 <= rows < 2**63:
        raise ValueError("root rows must be positive bounded integers")
    width = law.width
    # Charge retained raw scalar objects as well as output and temporary arrays.
    limits.admit_dimensions(
        (width,),
        ((),),
        rows=rows,
        calibration_rows=0,
        features=width,
        outputs=width,
        extra_parameters=8 * width
        + sum(len(x.probabilities) for x in law.coordinates)
        + rows * width,
        extra_working_channels=4 * width,
    )
    if capture is not None and type(capture) is not RawDrawLedger:
        raise ValueError("root capture must be a raw draw ledger")
    ledger = capture if capture is not None else RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        output = np.empty((rows, width), dtype=np.float64)
        for column, coordinate in enumerate(law.coordinates):
            family = coordinate.family
            distribution = (
                "normal"
                if family == "gaussian"
                else "choice"
                if family == "categorical"
                else "zipf"
                if family == "zipf"
                else "random"
            )
            parameters = (
                (
                    ("count", len(coordinate.probabilities)),
                    ("probabilities", coordinate.probabilities),
                )
                if family == "categorical"
                else (("exponent", coordinate.exponent),)
                if family == "zipf"
                else ()
            )
            raw = staged.draw(
                stream,
                owner=(*owner, f"column:{column}"),
                role="values",
                distribution=distribution,
                shape=(rows,),
                parameters=parameters,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                if family == "gaussian":
                    values = coordinate.location + coordinate.scale * raw
                elif family == "symmetric_uniform":
                    values = -1 + 2 * raw
                elif family == "zipf" and coordinate.clip is not None:
                    values = np.minimum(raw, coordinate.clip)
                else:
                    values = raw
                output[:, column] = values
        if not np.isfinite(output).all():
            raise ValueError("nonfinite source root emission; no retry")
    return output
