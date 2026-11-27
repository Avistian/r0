"""Explicit raw-draw capture staged atomically with each construction RNG."""

import copy
import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.joint_sampling_records import validate_name
from r0.modules.priors.lib.joint_streams import JointStreamLimits, stream_states
from r0.modules.priors.lib.raw_draw_records import (
    RawDrawLimits,
    RawDrawRecord,
    numpy_draw,
    state_digest,
)
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec

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
        self._bindings: dict[str, np.random.Generator] = {}
        self._expected: dict[str, str] = {}

    @property
    def limits(self) -> RawDrawLimits:
        return self._limits

    def _available(self) -> None:
        if self._busy or self._closed:
            raise ValueError("raw draw ledger is busy or closed")

    def freeze(self) -> tuple[RawDrawRecord, ...]:
        self._available()
        return tuple(self._records)

    def admit_limits(self, limits: RawDrawLimits) -> None:
        """Recheck already charged records against final enclosing construction caps."""
        self._available()
        if type(limits) is not RawDrawLimits:
            raise ValueError("raw admission requires frozen limits")
        if (
            len(self._records) > limits.draws
            or self._numbers > limits.numbers
            or self._bytes + 2 * limits.state_bytes > limits.working_bytes
        ):
            raise ValueError("final aggregate raw record limit exceeded")
        for rng in self._bindings.values():
            state_digest(rng, limits.state_bytes)

    def _admit_additional(self, draws: int, numbers: int, retained_bytes: int) -> None:
        if (
            len(self._records) + draws > self.limits.draws
            or self._numbers + numbers > self.limits.numbers
            or self._bytes + retained_bytes + 2 * self.limits.state_bytes
            > self.limits.working_bytes
        ):
            raise ValueError("raw draw retained record limit exceeded")

    def admit_draws(self, rng: np.random.Generator, draws: tuple[RawDrawSpec, ...]) -> None:
        """Admit a complete sequence without drawing, reserving or changing the ledger."""
        self._available()
        if type(draws) is not tuple or any(type(draw) is not RawDrawSpec for draw in draws):
            raise ValueError("raw admission requires frozen draw declarations")
        stream = self._stream_for(rng) if self._staged else "raw"
        before = state_digest(rng, self.limits.state_bytes)
        if self._staged and before != self._expected[stream]:
            raise ValueError("unrecorded RNG draw before raw admission")
        self._admit_additional(
            len(draws),
            sum(draw.numbers for draw in draws),
            sum(draw.retained_bytes(stream) for draw in draws),
        )

    def _stream_for(self, rng: np.random.Generator) -> str:
        for name, bound in self._bindings.items():
            if rng is bound:
                return name
        raise ValueError("raw draw requires its bound transaction RNG")

    @contextmanager
    def transaction(
        self, rng: np.random.Generator
    ) -> Generator[tuple[np.random.Generator, "RawDrawLedger"]]:
        self._available()
        role = self._stream_for(rng) if self._staged else "raw"
        with self.transactions({role: rng}) as (streams, ledger):
            yield streams[role], ledger

    @contextmanager
    def transactions(
        self, streams: dict[str, np.random.Generator]
    ) -> Generator[tuple[dict[str, np.random.Generator], "RawDrawLedger"]]:
        self._available()
        validate_raw_streams(streams)
        bindings = streams.copy()
        if self._staged and any(
            self._bindings.get(name) is not rng for name, rng in streams.items()
        ):
            raise ValueError("nested capture requires its bound transaction RNG names")
        before = raw_stream_states(streams, self.limits)
        if self._staged and any(before[name] != self._expected[name] for name in streams):
            raise ValueError("unrecorded RNG draw before nested capture")
        if self._bytes + 2 * self.limits.state_bytes > self.limits.working_bytes:
            raise ValueError("raw draw working byte limit exceeded")
        staged_streams = copy.deepcopy(streams)
        staged = RawDrawLedger(limits=self.limits)
        staged._records = self._records.copy()
        staged._numbers, staged._bytes = self._numbers, self._bytes
        staged._staged = True
        staged._bindings = staged_streams.copy()
        staged._expected = before.copy()
        self._busy = True
        try:
            yield staged_streams, staged
            staged._available()
            validate_stream_bindings(streams, bindings)
            validate_stream_bindings(staged_streams, staged._bindings)
            if raw_stream_states(streams, self.limits) != before:
                raise ValueError("caller RNG changed during raw draw transaction")
            if raw_stream_states(staged_streams, self.limits) != staged._expected:
                raise ValueError("unrecorded RNG draw during raw capture")
            # Prepare all fallible copies before any caller state or ledger changes.
            states = {name: rng.bit_generator.state for name, rng in staged_streams.items()}
            records = staged._records.copy()
            expected = {**self._expected, **staged._expected}
            for name, rng in streams.items():
                rng.bit_generator.state = states[name]
            self._records = records
            self._numbers, self._bytes = staged._numbers, staged._bytes
            self._expected = expected
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
        stream: str | None = None,
    ) -> NDArray[Any]:
        self._available()
        if not self._staged:
            raise ValueError("raw draws require an active staged ledger transaction")
        bound = self._stream_for(rng)
        if stream is not None and stream != bound:
            raise ValueError("raw draw stream label differs from its bound RNG")
        stream = bound
        spec = RawDrawSpec(owner, role, distribution, shape, parameters)
        count, cost = spec.numbers, spec.retained_bytes(stream)
        self._admit_additional(1, count, cost)
        before = state_digest(rng, self.limits.state_bytes)
        if before != self._expected[stream]:
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
        self._expected[stream] = record.state_after
        self._numbers += count
        self._bytes += cost
        return values


def validate_raw_streams(streams: dict[str, np.random.Generator]) -> None:
    if type(streams) is not dict or not 1 <= len(streams) <= 64:
        raise ValueError("raw capture requires one to 64 named streams")
    identities: set[int] = set()
    for name, rng in streams.items():
        validate_name(name, "raw stream name")
        state_digest(rng)
        identity = id(rng.bit_generator)
        if identity in identities:
            raise ValueError("raw capture streams must have distinct bit generators")
        identities.add(identity)


def raw_stream_states(
    streams: dict[str, np.random.Generator], limits: RawDrawLimits
) -> dict[str, str]:
    states = stream_states(
        tuple(streams.values()), JointStreamLimits(state_bytes=limits.state_bytes)
    )
    return {
        name: hashlib.sha256(state).hexdigest() for name, state in zip(streams, states, strict=True)
    }


def validate_stream_bindings(
    streams: dict[str, np.random.Generator], bindings: dict[str, np.random.Generator]
) -> None:
    if streams.keys() != bindings.keys() or any(
        streams[name] is not rng for name, rng in bindings.items()
    ):
        raise ValueError("raw transaction stream bindings changed")


@contextmanager
def raw_transactions(
    streams: dict[str, np.random.Generator], capture: RawDrawLedger | None
) -> Generator[tuple[dict[str, np.random.Generator], RawDrawLedger | None]]:
    if capture is None:
        validate_raw_streams(streams)
        bindings = streams.copy()
        before = raw_stream_states(streams, _DEFAULT_LIMITS)
        staged = copy.deepcopy(streams)
        staged_bindings = staged.copy()
        yield staged, None
        validate_stream_bindings(streams, bindings)
        validate_stream_bindings(staged, staged_bindings)
        if raw_stream_states(streams, _DEFAULT_LIMITS) != before:
            raise ValueError("caller RNG changed during construction")
        states = {name: rng.bit_generator.state for name, rng in staged.items()}
        for name, rng in streams.items():
            rng.bit_generator.state = states[name]
    elif type(capture) is RawDrawLedger:
        with capture.transactions(streams) as pair:
            yield pair
    else:
        raise ValueError("raw capture must be a RawDrawLedger")


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
