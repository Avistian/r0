"""Bounded JSON snapshots of the five supported NumPy generators; never unpickle."""

import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.raw_draw_records import state_digest

_ENGINES = {
    name: getattr(np.random, name)
    for name in (
        "PCG64",
        "PCG64DXSM",
        "Philox",
        "SFC64",
        "MT19937",
    )
}
_DEFAULT_BYTES = 1024**2


def _array(value: Any) -> dict[str, Any]:
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.str not in ("<u4", "<u8"):
        raise ValueError("unsupported generator state array")
    return {"array_dtype": value.dtype.str, "values": value.tolist()}


def _restore_array(value: dict[str, Any]) -> Any:
    if "array_dtype" not in value:
        return value
    if set(value) != {"array_dtype", "values"} or value["array_dtype"] not in ("<u4", "<u8"):
        raise ValueError("invalid generator array descriptor")
    if type(value["values"]) is not list:
        raise ValueError("invalid generator state array values")
    values = cast(list[Any], value["values"])
    bits = 32 if value["array_dtype"] == "<u4" else 64
    if len(values) > _DEFAULT_BYTES // 4 or any(
        type(v) is not int or not 0 <= v < 2**bits for v in values
    ):
        raise ValueError("invalid generator state array values")
    return np.asarray(values, dtype=value["array_dtype"])


def _payload(rng: np.random.Generator) -> str:
    return json.dumps(
        rng.bit_generator.state, default=_array, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True, slots=True)
class GeneratorSnapshot:
    engine: str
    payload: str
    digest: str
    numpy_version: str
    version: str = "numpy-generator-json-v1"

    def restore(self, *, byte_limit: int = _DEFAULT_BYTES) -> np.random.Generator:
        if (
            self.version != "numpy-generator-json-v1"
            or type(self.engine) is not str
            or self.engine not in _ENGINES
            or self.numpy_version != np.__version__
            or type(self.payload) is not str
            or type(byte_limit) is not int
            or not 0 < len(self.payload) <= min(byte_limit, _DEFAULT_BYTES)
            or not self.payload.isascii()
        ):
            raise ValueError("unsupported generator snapshot version, engine, environment or size")
        try:
            state = json.loads(self.payload, object_hook=_restore_array)
            rng = np.random.Generator(_ENGINES[self.engine](0))
            rng.bit_generator.state = state
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise ValueError("invalid generator state snapshot") from exc
        if _payload(rng) != self.payload or state_digest(rng, byte_limit) != self.digest:
            raise ValueError("generator snapshot is not canonical or differs from its digest")
        return rng


def snapshot_generator(
    rng: np.random.Generator, *, byte_limit: int = _DEFAULT_BYTES
) -> GeneratorSnapshot:
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError("generator snapshot byte limit must be positive")
    digest = state_digest(rng, byte_limit)
    payload = _payload(rng)
    if len(payload) > min(byte_limit, _DEFAULT_BYTES):
        raise ValueError("generator snapshot byte limit exceeded")
    return GeneratorSnapshot(type(rng.bit_generator).__name__, payload, digest, np.__version__)
