"""Validate five supported retained NumPy state schemas without instantiating an RNG."""

import hashlib
import json
import pickle
from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot


def _object(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict or tuple(cast(dict[str, Any], value)) != keys:
        raise ValueError("joint snapshot requires the canonical state field schema")
    return cast(dict[str, Any], value)


def _integer(value: Any, upper: int) -> int:
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError("joint snapshot state requires bounded exact integer values")
    return value


def _array(value: Any, bits: int, count: int) -> Any:
    raw = _object(value, ("array_dtype", "values"))
    values = raw["values"]
    if (
        raw["array_dtype"] != f"<u{bits // 8}"
        or type(values) is not list
        or len(cast(list[Any], values)) != count
    ):
        raise ValueError("joint snapshot array differs from declared engine geometry")
    for item in cast(list[Any], values):
        _integer(item, 2**bits)
    return np.asarray(cast(list[int], values), dtype=cast(str, raw["array_dtype"]))


def validate_joint_snapshot(snapshot: GeneratorSnapshot, *, byte_limit: int) -> None:
    """Only admitted fixed-size state arrays are created; no restore or random draw occurs."""
    snapshot.validate_metadata(byte_limit=byte_limit)
    engine = snapshot.engine
    fields = (
        ("bit_generator", "state")
        if engine == "MT19937"
        else ("bit_generator", "state", "buffer", "buffer_pos", "has_uint32", "uinteger")
        if engine == "Philox"
        else ("bit_generator", "state", "has_uint32", "uinteger")
    )
    state = _object(json.loads(snapshot.payload), fields)
    if json.dumps(state, separators=(",", ":"), allow_nan=False) != snapshot.payload:
        raise ValueError("joint snapshot payload is not canonical JSON")
    if engine in ("PCG64", "PCG64DXSM"):
        inner = _object(state["state"], ("state", "inc"))
        _integer(inner["state"], 2**128)
        _integer(inner["inc"], 2**128)
    elif engine == "SFC64":
        inner = _object(state["state"], ("state",))
        inner["state"] = _array(inner["state"], 64, 4)
    elif engine == "Philox":
        inner = _object(state["state"], ("counter", "key"))
        inner["counter"] = _array(inner["counter"], 64, 4)
        inner["key"] = _array(inner["key"], 64, 2)
        state["buffer"] = _array(state["buffer"], 64, 4)
        _integer(state["buffer_pos"], 5)
    else:
        inner = _object(state["state"], ("key", "pos"))
        inner["key"] = _array(inner["key"], 32, 624)
        _integer(inner["pos"], 625)
    if engine != "MT19937":
        _integer(state["has_uint32"], 2)
        _integer(state["uinteger"], 2**32)
    # Match state_digest's existing trusted live-state encoding. No pickle is read.
    encoded = pickle.dumps(state)
    if len(encoded) > byte_limit or hashlib.sha256(encoded).hexdigest() != snapshot.digest:
        raise ValueError("joint snapshot state differs from retained digest")
