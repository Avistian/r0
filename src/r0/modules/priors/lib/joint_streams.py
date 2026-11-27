"""Validate and bound caller-injected streams before transactional state copies."""

import pickle
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.joint_sampling_records import validate_name


@dataclass(frozen=True, slots=True)
class JointStreamLimits:
    """Auxiliary stream count and serialized state bytes, including the named-choice RNG."""

    count: int = 64
    state_bytes: int = 1024**2

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or not 1 <= value < 2**63
            for value in (self.count, self.state_bytes)
        ):
            raise ValueError("joint stream limits must be positive bounded integers")


def stream_states(
    generators: tuple[np.random.Generator, ...], limits: JointStreamLimits
) -> tuple[bytes, ...]:
    """Serialize only trusted live NumPy states for comparison; never unpickle inputs."""
    states: list[bytes] = []
    size = 0
    for generator in generators:
        state = pickle.dumps(generator.bit_generator.state)
        size += len(state)
        if size > limits.state_bytes:
            raise ValueError("joint stream state byte limit exceeded")
        states.append(state)
    return tuple(states)


def validate_streams(
    named: np.random.Generator,
    streams: dict[str, np.random.Generator] | None,
    limits: JointStreamLimits,
) -> dict[str, np.random.Generator]:
    if type(named) is not np.random.Generator or type(limits) is not JointStreamLimits:
        raise ValueError("joint streams require injected NumPy generators and frozen limits")
    if streams is None:
        streams = {}
    if type(streams) is not dict:
        raise ValueError("joint streams require a dictionary of named injected generators")
    if len(streams) > limits.count:
        raise ValueError("joint stream count limit exceeded")
    identities = {id(named.bit_generator)}
    for name, generator in streams.items():
        validate_name(name, "stream name")
        if type(generator) is not np.random.Generator:
            raise ValueError("joint streams require injected NumPy generators")
        identity = id(generator.bit_generator)
        if identity in identities:
            raise ValueError("joint named and auxiliary streams require distinct bit generators")
        identities.add(identity)
    stream_states((named, *streams.values()), limits)
    return dict(streams)
