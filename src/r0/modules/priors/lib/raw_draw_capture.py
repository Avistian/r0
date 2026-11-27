"""Explicit raw-draw capture staged atomically with each construction RNG."""

import copy
import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.joint_sampling_records import validate_name, validate_path
from r0.modules.priors.lib.raw_draw_records import (
    RawDrawLimits,
    RawDrawRecord,
    draw_arguments,
    numpy_draw,
    state_digest,
)

_DEFAULT_LIMITS = RawDrawLimits()


class RawDrawLedger:
    def __init__(self, *, limits: RawDrawLimits = _DEFAULT_LIMITS) -> None:
        if type(limits) is not RawDrawLimits:
            raise ValueError("raw draw ledger requires frozen limits")
        self._limits = limits
        self._records: list[RawDrawRecord] = []
        self._numbers = 0
        self._bytes = 0
        self._busy = False
        self._closed = False
        self._staged = False
        self._rng: np.random.Generator | None = None
        self._expected_state: str | None = None

    @property
    def limits(self) -> RawDrawLimits:
        return self._limits

    def _available(self) -> None:
        if self._busy or self._closed:
            raise ValueError("raw draw ledger is busy or closed")

    def freeze(self) -> tuple[RawDrawRecord, ...]:
        self._available()
        return tuple(self._records)

    @contextmanager
    def transaction(
        self, rng: np.random.Generator
    ) -> Generator[tuple[np.random.Generator, "RawDrawLedger"]]:
        self._available()
        if self._staged and rng is not self._rng:
            raise ValueError("nested capture requires its bound transaction RNG")
        before = state_digest(rng, self.limits.state_bytes)
        if self._staged and before != self._expected_state:
            raise ValueError("unrecorded RNG draw before nested capture")
        if self._bytes + 2 * self.limits.state_bytes > self.limits.working_bytes:
            raise ValueError("raw draw working byte limit exceeded")
        staged_rng = copy.deepcopy(rng)
        staged = RawDrawLedger(limits=self.limits)
        staged._records = self._records.copy()
        staged._numbers, staged._bytes = self._numbers, self._bytes
        staged._staged = True
        staged._rng = staged_rng
        staged._expected_state = before
        self._busy = True
        try:
            yield staged_rng, staged
            staged._available()
            if state_digest(rng, self.limits.state_bytes) != before:
                raise ValueError("caller RNG changed during raw draw transaction")
            if state_digest(staged_rng, self.limits.state_bytes) != staged._expected_state:
                raise ValueError("unrecorded RNG draw during raw capture")
            state = staged_rng.bit_generator.state
            # Prepare every fallible record/state copy before committing caller state.
            records = staged._records.copy()
            rng.bit_generator.state = state
            self._records = records
            self._numbers, self._bytes = staged._numbers, staged._bytes
            self._expected_state = staged._expected_state
        finally:
            self._busy = False
            staged._closed = True

    def draw(
        self,
        rng: np.random.Generator,
        *,
        owner: tuple[str, ...],
        role: str,
        distribution: str,
        shape: tuple[int, ...],
        parameters: tuple[tuple[str, Any], ...] = (),
        stream: str = "raw",
    ) -> NDArray[Any]:
        self._available()
        if not self._staged:
            raise ValueError("raw draws require an active staged ledger transaction")
        if rng is not self._rng:
            raise ValueError("raw draw requires its bound transaction RNG")
        validate_path(owner)
        validate_name(role, "raw draw role")
        validate_name(stream, "raw draw stream")
        draw_arguments(distribution, shape, parameters)
        count = math.prod(shape)
        metadata = sum(
            len(cast(tuple[Any, ...], v)) if type(v) is tuple else 1 for _, v in parameters
        )
        cost = (
            96 * count
            + 128 * (metadata + len(shape) + len(owner))
            + 1024
            + sum(len(s.encode()) for s in (*owner, role, stream, distribution))
        )
        if (
            len(self._records) + 1 > self.limits.draws
            or self._numbers + count > self.limits.numbers
            or self._bytes + cost + 2 * self.limits.state_bytes > self.limits.working_bytes
        ):
            raise ValueError("raw draw retained record limit exceeded")
        before = state_digest(rng, self.limits.state_bytes)
        if before != self._expected_state:
            raise ValueError("unrecorded RNG draw before raw capture")
        values = numpy_draw(rng, distribution, shape, parameters)
        integer = values.dtype.kind in "iu"
        record = RawDrawRecord(
            owner,
            role,
            stream,
            len(self._records),
            distribution,
            shape,
            parameters,
            "<i8" if integer else "<f8",
            tuple(map(int if integer else float, values.flat)),
            type(rng.bit_generator).__name__,
            before,
            state_digest(rng, self.limits.state_bytes),
        )
        self._records.append(record)
        self._expected_state = record.state_after
        self._numbers += count
        self._bytes += cost
        return values


@contextmanager
def raw_transaction(
    rng: np.random.Generator, capture: RawDrawLedger | None
) -> Generator[tuple[np.random.Generator, RawDrawLedger | None]]:
    if capture is None:
        staged = copy.deepcopy(rng)
        yield staged, None
        rng.bit_generator.state = staged.bit_generator.state
    elif type(capture) is RawDrawLedger:
        with capture.transaction(rng) as pair:
            yield pair
    else:
        raise ValueError("raw capture must be a RawDrawLedger")


def recorded_draw(
    rng: np.random.Generator,
    capture: RawDrawLedger | None,
    *,
    owner: tuple[str, ...],
    role: str,
    distribution: str,
    shape: tuple[int, ...],
    parameters: tuple[tuple[str, Any], ...] = (),
) -> NDArray[Any]:
    if capture is None:
        return numpy_draw(rng, distribution, shape, parameters)
    return capture.draw(
        rng, owner=owner, role=role, distribution=distribution, shape=shape, parameters=parameters
    )
