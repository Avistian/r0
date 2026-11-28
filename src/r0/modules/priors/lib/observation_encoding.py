"""Explicit finite-value discretization and count observation schemas."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.training_features import TrainingFeature

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class ColumnEncoding:
    column: int
    kind: Literal["binary", "categorical", "ordinal", "count"]
    cutpoints: tuple[float, ...] = ()
    vocabulary: tuple[str, ...] = ()
    equality: str = "lower"
    count_rounding: str = "floor"
    count_scale: float = 1.0
    version: str = "r0-column-encoding-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-column-encoding-v1"
            or type(self.column) is not int
            or not 0 <= self.column < 256
            or self.kind not in ("binary", "categorical", "ordinal", "count")
            or type(self.cutpoints) is not tuple
            or len(self.cutpoints) > 1023
            or any(not finite_number(v) for v in self.cutpoints)
            or any(a >= b for a, b in zip(self.cutpoints, self.cutpoints[1:], strict=False))
            or type(self.vocabulary) is not tuple
            or any(
                type(label) is not str or not 1 <= len(label) <= 256 for label in self.vocabulary
            )
            or self.equality not in ("lower", "upper")
            or self.count_rounding not in ("floor", "nearest", "ceil")
            or not finite_number(self.count_scale)
            or self.count_scale <= 0
        ):
            raise ValueError("invalid bounded column encoding")
        if self.kind == "count":
            if self.cutpoints or self.vocabulary or self.equality != "lower":
                raise ValueError("count encoding does not use categories or threshold equality")
        elif (
            len(self.vocabulary) != len(self.cutpoints) + 1
            or self.count_rounding != "floor"
            or self.count_scale != 1
        ):
            raise ValueError("encoding vocabulary must match bins; count controls are inactive")
        TrainingFeature(self.kind, self.vocabulary)


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    encodings: tuple[ColumnEncoding, ...]
    values: NDArray[np.float64]
    schema: tuple[TrainingFeature, ...]
    binding: str
    version: str = "r0-encoded-observation-v1"


def encode_observation_columns(
    encodings: tuple[ColumnEncoding, ...],
    inputs: NDArray[np.float64],
    *,
    schema: tuple[TrainingFeature, ...] = (),
    limits: CausalLimits = _DEFAULT,
) -> EncodedObservation:
    """Discretize with fixed external thresholds; never fit on context/query rows.

    Count encoding rounds max(0, x/count_scale), with nearest using ties-to-even.
    Category bin order follows cutpoints and explicit labels, not lexical sorting.
    Each column may be encoded once; noninvertibility is part of the declared map.
    """
    if (
        type(encodings) is not tuple
        or not 1 <= len(encodings) <= 256
        or any(type(e) is not ColumnEncoding for e in encodings)
        or len({e.column for e in encodings}) != len(encodings)
        or type(inputs) is not np.ndarray
        or inputs.dtype != np.float64
        or inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 20000
        or not 1 <= inputs.shape[1] <= 256
        or type(schema) is not tuple
        or any(type(f) is not TrainingFeature for f in schema)
        or schema
        and len(schema) != inputs.shape[1]
        or type(limits) is not CausalLimits
    ):
        raise ValueError("encoding requires bounded mappings, float64 inputs and matching schema")
    if any(e.column >= inputs.shape[1] for e in encodings):
        raise ValueError("encoding column lies outside schema")
    _, storage = function_record_cost((encodings, schema), limits)
    if 65536 + 64 * inputs.size + 4 * storage > limits.working_bytes:
        raise ValueError("encoding workspace exceeds limits")
    if not np.isfinite(inputs).all():
        raise ValueError("encoding requires finite values before masking")
    original = schema or (TrainingFeature(),) * inputs.shape[1]
    for encoding in encodings:
        if original[encoding.column].kind not in ("continuous", "count"):
            raise ValueError(
                "encoding requires numeric source columns; category relabeling is separate"
            )
    digest = hashlib.sha256(
        json.dumps(
            {"encodings": [asdict(e) for e in encodings], "schema": [asdict(f) for f in original]},
            sort_keys=True,
        ).encode()
    )
    digest.update(str(inputs.shape).encode())
    digest.update(np.ascontiguousarray(inputs, dtype="<f8").tobytes())
    values = inputs.copy()
    output_schema = list(original)
    for encoding in encodings:
        x = inputs[:, encoding.column]
        if encoding.kind == "count":
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = np.maximum(0, x / encoding.count_scale)
                output = {"floor": np.floor, "nearest": np.rint, "ceil": np.ceil}[
                    encoding.count_rounding
                ](scaled)
            if not np.isfinite(output).all():
                raise ValueError("count encoding overflow; no hidden saturation")
        else:
            output = np.searchsorted(
                np.asarray(encoding.cutpoints),
                x,
                side="left" if encoding.equality == "lower" else "right",
            )
        values[:, encoding.column] = output
        output_schema[encoding.column] = TrainingFeature(encoding.kind, encoding.vocabulary)
    return EncodedObservation(encodings, values, tuple(output_schema), digest.hexdigest())
