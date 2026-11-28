"""Declared encodings for numeric training buffers containing native variables."""

import math
from dataclasses import dataclass
from typing import Literal

from r0.modules.episodes import Value


@dataclass(frozen=True, slots=True)
class TrainingFeature:
    kind: Literal["continuous", "count", "binary", "categorical", "ordinal"] = "continuous"
    vocabulary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "count", "binary", "categorical", "ordinal"):
            raise ValueError("unsupported training feature kind")
        if type(self.vocabulary) is not tuple or any(
            type(v) is not str or not v for v in self.vocabulary
        ):
            raise ValueError("training feature vocabulary must contain declared labels")
        if len(set(self.vocabulary)) != len(self.vocabulary) or (
            self.kind in ("continuous", "count")
            and self.vocabulary
            or self.kind == "binary"
            and len(self.vocabulary) != 2
            or self.kind in ("categorical", "ordinal")
            and len(self.vocabulary) < 1
        ):
            raise ValueError("training feature vocabulary does not match its kind")

    def decode(self, value: float) -> Value:
        if self.kind in ("continuous", "count"):
            if not math.isfinite(value):
                raise ValueError("observed numeric feature must be finite")
            if self.kind == "count":
                if value < 0 or value != int(value):
                    raise ValueError("count feature must be a nonnegative integer")
                return int(value)
            return float(value)
        if not math.isfinite(value) or value != int(value) or not 0 <= value < len(self.vocabulary):
            raise ValueError("native feature code lies outside its declared vocabulary")
        return int(value) if self.kind == "binary" else self.vocabulary[int(value)]
