"""Frozen raw draw declarations and their exact retained-record admission cost."""

import math
from dataclasses import dataclass
from typing import Any, cast

from r0.modules.priors.lib.joint_sampling_records import validate_name, validate_path
from r0.modules.priors.lib.raw_draw_records import draw_arguments


@dataclass(frozen=True, slots=True)
class RawDrawSpec:
    owner: tuple[str, ...]
    role: str
    distribution: str
    shape: tuple[int, ...]
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        validate_path(self.owner)
        validate_name(self.role, "raw draw role")
        draw_arguments(self.distribution, self.shape, self.parameters)

    @property
    def numbers(self) -> int:
        return math.prod(self.shape)

    def retained_bytes(self, stream: str) -> int:
        validate_name(stream, "raw draw stream")
        metadata = sum(
            len(cast(tuple[Any, ...], v)) if type(v) is tuple else 1 for _, v in self.parameters
        )
        return (
            96 * self.numbers
            + 128 * (metadata + len(self.shape) + len(self.owner))
            + 1024
            + sum(len(s.encode()) for s in (*self.owner, self.role, stream, self.distribution))
        )
