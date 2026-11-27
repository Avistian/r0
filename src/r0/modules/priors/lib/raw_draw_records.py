"""Lossless raw draw records with explicit laws and live-state replay checks."""

import copy
import hashlib
import math
import pickle
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.joint_sampling_records import validate_name, validate_path


@dataclass(frozen=True, slots=True)
class RawDrawLimits:
    draws: int = 100_000
    numbers: int = 1_000_000
    working_bytes: int = 256 * 1024**2
    state_bytes: int = 1024**2

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or not 1 <= v < 2**63
            for v in (
                self.draws,
                self.numbers,
                self.working_bytes,
                self.state_bytes,
            )
        ):
            raise ValueError("raw draw limits must be positive bounded integers")


def state_digest(rng: np.random.Generator, limit: int = 1024**2) -> str:
    """Hash trusted live state only. No record ever contains or unpickles a pickle."""
    # State publication relies on the concrete NumPy setter contract. A subclass
    # may impersonate an engine name while overriding setters or deepcopy and
    # cause a partially published multi-stream transaction.
    if type(rng) is not np.random.Generator or type(rng.bit_generator) not in (
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.Philox,
        np.random.SFC64,
        np.random.MT19937,
    ):
        raise ValueError("raw draws require a supported injected NumPy generator")
    raw = pickle.dumps(rng.bit_generator.state)
    if len(raw) > limit:
        raise ValueError("raw draw RNG state byte limit exceeded")
    return hashlib.sha256(raw).hexdigest()


def draw_arguments(
    distribution: str, shape: tuple[int, ...], parameters: tuple[tuple[str, Any], ...]
) -> dict[str, Any]:
    if (
        type(shape) is not tuple
        or len(shape) > 8
        or any(type(n) is not int or not 1 <= n < 2**63 for n in shape)
    ):
        raise ValueError("raw draw shape must contain bounded positive dimensions")
    if type(parameters) is not tuple or any(
        type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str for pair in parameters
    ):
        raise ValueError("raw draw parameters must be frozen named pairs")
    args = dict(parameters)
    if len(args) != len(parameters):
        raise ValueError("raw draw parameters must have distinct names")
    expected: dict[str, set[str]] = {
        "normal": set(),
        "cauchy": set(),
        "random": set(),
        "uniform": {"low", "high"},
        "student_t": {"df"},
        "gamma": {"concentration"},
        "integers": {"high"},
        "permutation": {"count"},
        "choice": {"count", "probabilities"},
    }
    if distribution not in expected or set(args) != expected[distribution]:
        raise ValueError("unsupported raw draw law or parameter fields")
    if distribution == "uniform" and (
        not finite_number(args["low"])
        or not finite_number(args["high"])
        or args["low"] >= args["high"]
    ):
        raise ValueError("raw uniform bounds must be finite and ordered")
    if distribution == "student_t" and (not finite_number(args["df"]) or args["df"] <= 0):
        raise ValueError("raw Student-t degrees must be positive and finite")
    if distribution == "gamma" and (
        not finite_number(args["concentration"]) or args["concentration"] <= 0
    ):
        raise ValueError("raw Gamma concentration must be finite and positive")
    if distribution in ("integers", "permutation", "choice"):
        bound = args["high" if distribution == "integers" else "count"]
        if type(bound) is not int or not 1 <= bound < 2**63:
            raise ValueError("raw discrete bounds must be positive bounded integers")
    if distribution == "permutation" and shape != (args["count"],):
        raise ValueError("raw permutation shape must match its count")
    if distribution == "choice":
        if type(args["probabilities"]) is not tuple:
            raise ValueError("raw choice probabilities must be a frozen vector")
        probabilities = cast(tuple[Any, ...], args["probabilities"])
        if len(probabilities) != args["count"] or any(
            not finite_number(v) or v < 0 for v in probabilities
        ):
            raise ValueError("raw choice probabilities must be a finite nonnegative vector")
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("raw choice probabilities must sum to one")
    return args


def numpy_draw(
    rng: np.random.Generator,
    distribution: str,
    shape: tuple[int, ...],
    parameters: tuple[tuple[str, Any], ...],
) -> NDArray[Any]:
    args = draw_arguments(distribution, shape, parameters)
    if distribution == "normal":
        values = rng.normal(size=shape)
    elif distribution == "cauchy":
        values = rng.standard_cauchy(size=shape)
    elif distribution == "random":
        values = rng.random(size=shape)
    elif distribution == "uniform":
        values = rng.uniform(args["low"], args["high"], size=shape)
    elif distribution == "student_t":
        values = rng.standard_t(args["df"], size=shape)
    elif distribution == "gamma":
        values = rng.gamma(args["concentration"], size=shape)
    elif distribution == "integers":
        values = rng.integers(args["high"], size=shape)
    elif distribution == "permutation":
        values = rng.permutation(args["count"])
    else:
        values = rng.choice(args["count"], size=shape, p=args["probabilities"])
    return np.asarray(values)


@dataclass(frozen=True, slots=True)
class RawDrawRecord:
    owner: tuple[str, ...]
    role: str
    stream: str
    sequence: int
    distribution: str
    shape: tuple[int, ...]
    parameters: tuple[tuple[str, Any], ...]
    dtype: str
    values: tuple[float | int, ...]
    generator: str
    state_before: str
    state_after: str
    version: str = "raw-draw-v1"

    def __post_init__(self) -> None:
        validate_path(self.owner)
        validate_name(self.role, "raw draw role")
        validate_name(self.stream, "raw draw stream")
        if self.version != "raw-draw-v1" or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("raw draw requires its version and nonnegative sequence")
        draw_arguments(self.distribution, self.shape, self.parameters)
        integer = self.distribution in ("integers", "permutation", "choice")
        if (
            self.dtype != ("<i8" if integer else "<f8")
            or type(self.values) is not tuple
            or len(self.values) != math.prod(self.shape)
        ):
            raise ValueError("raw draw values must match canonical dtype and shape")
        if any(
            not finite_number(v) or integer and (type(v) is not int or not -(2**63) <= v < 2**63)
            for v in self.values
        ):
            raise ValueError("raw draw values must be finite and match their numeric type")
        if self.generator not in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937") or any(
            type(s) is not str or re.fullmatch(r"[0-9a-f]{64}", s) is None
            for s in (self.state_before, self.state_after)
        ):
            raise ValueError("raw draw requires supported generator and state digests")

    def array(self) -> NDArray[Any]:
        return np.asarray(self.values, dtype=self.dtype).reshape(self.shape)

    def replay(self, rng: np.random.Generator) -> NDArray[Any]:
        """Caller supplies the matching live pre-state; a digest does not recover a seed."""
        if (
            type(rng) is not np.random.Generator
            or type(rng.bit_generator).__name__ != self.generator
            or state_digest(rng) != self.state_before
        ):
            raise ValueError("raw draw replay requires its original generator pre-state")
        staged = copy.deepcopy(rng)
        result = numpy_draw(staged, self.distribution, self.shape, self.parameters)
        if (
            state_digest(staged) != self.state_after
            or np.asarray(result, dtype=self.dtype).tobytes() != self.array().tobytes()
        ):
            raise ValueError("raw draw replay differs from retained values or state")
        if state_digest(rng) != self.state_before:
            raise ValueError("caller RNG changed during raw draw replay")
        rng.bit_generator.state = staged.bit_generator.state
        return result
