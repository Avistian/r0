"""Bounded r0 missingness mechanisms; masks do not alter structural equations."""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import raw_transactions

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class MissingnessPlan:
    mode: str
    columns: tuple[int, ...]
    probability: float | None = None
    drivers: tuple[int, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float = 0.0
    slope: float = 1.0
    block_rows: int = 1
    version: str = "r0-missingness-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-missingness-plan-v1"
            or self.mode
            not in ("mcar", "mar", "mnar-self", "mnar-target", "mnar-latent", "row", "block")
            or type(self.columns) is not tuple
            or not 1 <= len(self.columns) <= 256
            or any(type(c) is not int or not 0 <= c < 256 for c in self.columns)
            or len(set(self.columns)) != len(self.columns)
            or self.probability is not None
            and (not finite_number(self.probability) or not 0 <= self.probability <= 1)
            or type(self.drivers) is not tuple
            or len(set(self.drivers)) != len(self.drivers)
            or any(type(c) is not int or not 0 <= c < 256 for c in self.drivers)
            or type(self.coefficients) is not tuple
            or any(not finite_number(v) for v in self.coefficients)
            or not finite_number(self.intercept)
            or not finite_number(self.slope)
            or type(self.block_rows) is not int
            or not 1 <= self.block_rows <= 20000
        ):
            raise ValueError("invalid bounded missingness plan")
        if self.mode == "mar":
            if (
                not self.drivers
                or len(self.drivers) != len(self.coefficients)
                or set(self.drivers) & set(self.columns)
            ):
                raise ValueError(
                    "MAR needs named, unmasked conditioning columns and matching coefficients"
                )
        elif self.drivers or self.coefficients:
            raise ValueError("conditioning columns apply only to MAR")
        if self.mode != "block" and self.block_rows != 1:
            raise ValueError("block_rows applies only to block missingness")
        logistic = self.mode in ("mar", "mnar-self", "mnar-target", "mnar-latent")
        if logistic and self.probability is not None:
            raise ValueError("logistic masks use intercept/slope, not fixed probability")
        if not logistic and self.probability is None:
            raise ValueError("fixed-probability masks require explicit probability")
        if not logistic and (self.intercept != 0 or self.slope != 1):
            raise ValueError("fixed-probability masks do not use logistic controls")
        if self.mode == "mar" and self.slope != 1:
            raise ValueError("MAR uses named coefficients, not scalar slope")


@dataclass(frozen=True, slots=True)
class MissingnessDraw:
    plan: MissingnessPlan
    mask: NDArray[np.bool_]
    probabilities: NDArray[np.float64]
    binding: str
    initial: GeneratorSnapshot
    final: GeneratorSnapshot


def _sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    # Avoid exp overflow without clipping the declared logits.
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1 + exp_value)
    return result


def draw_missingness(
    plan: MissingnessPlan,
    inputs: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    target: NDArray[np.float64] | None = None,
    latent: NDArray[np.float64] | None = None,
    limits: CausalLimits = _DEFAULT,
) -> MissingnessDraw:
    """Draw on finite pre-observation values; return full-width masks/probabilities.

    MAR drivers must be outside the masked columns and remain observed when this
    stage is composed. MNAR target/latent drivers are privileged caller inputs;
    their provenance must be retained by the enclosing observation task.
    """
    if (
        type(plan) is not MissingnessPlan
        or type(limits) is not CausalLimits
        or type(inputs) is not np.ndarray
        or inputs.dtype != np.float64
        or inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 20000
        or not 1 <= inputs.shape[1] <= 256
    ):
        raise ValueError("missingness requires bounded float64 input matrix and frozen controls")
    rows, width = inputs.shape
    if max(plan.columns + plan.drivers) >= width:
        raise ValueError("missingness column lies outside input schema")
    if 65536 + rows * width * 128 > limits.working_bytes:
        raise ValueError("missingness workspace exceeds limits")
    if not np.isfinite(inputs).all():
        raise ValueError("missingness source values must be finite")
    driver = (
        target if plan.mode == "mnar-target" else latent if plan.mode == "mnar-latent" else None
    )
    if (target is not None and plan.mode != "mnar-target") or (
        latent is not None and plan.mode != "mnar-latent"
    ):
        raise ValueError("unused privileged missingness driver")
    if plan.mode in ("mnar-target", "mnar-latent") and (
        type(driver) is not np.ndarray
        or driver.dtype != np.float64
        or driver.shape != (rows,)
        or not np.isfinite(driver).all()
    ):
        raise ValueError("MNAR requires a finite float64 row driver")
    import json

    digest = hashlib.sha256(json.dumps(asdict(plan), sort_keys=True).encode())
    digest.update(str(inputs.shape).encode())
    digest.update(np.ascontiguousarray(inputs, dtype="<f8").tobytes())
    if driver is not None:
        digest.update(np.ascontiguousarray(driver, dtype="<f8").tobytes())
    count = len(plan.columns)
    with raw_transactions({"missingness": rng}, None) as (streams, _):
        active = streams["missingness"]
        initial = snapshot_generator(active)
        if plan.mode in ("mcar", "row", "block"):
            assert plan.probability is not None
            selected = np.full((rows, count), plan.probability, dtype=np.float64)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                if plan.mode == "mar":
                    logits = (inputs[:, plan.drivers] @ np.asarray(plan.coefficients))[:, None]
                elif plan.mode == "mnar-self":
                    logits = plan.slope * inputs[:, plan.columns]
                else:
                    assert driver is not None
                    logits = plan.slope * driver[:, None]
                logits = logits + plan.intercept
            if not np.isfinite(logits).all():
                raise ValueError("missingness logits overflow; no silent clipping")
            selected = np.broadcast_to(_sigmoid(logits), (rows, count)).copy()
        if plan.mode == "row":
            draws = np.broadcast_to(active.random((rows, 1)), (rows, count))
        elif plan.mode == "block":
            block_count = math.ceil(rows / plan.block_rows)
            blocks = active.random((block_count, count))
            draws = blocks[np.arange(rows) // plan.block_rows]
        else:
            draws = active.random((rows, count))
        mask = np.zeros((rows, width), dtype=np.bool_)
        probabilities = np.zeros((rows, width), dtype=np.float64)
        mask[:, plan.columns] = draws < selected
        probabilities[:, plan.columns] = selected
        return MissingnessDraw(
            plan, mask, probabilities, digest.hexdigest(), initial, snapshot_generator(active)
        )
