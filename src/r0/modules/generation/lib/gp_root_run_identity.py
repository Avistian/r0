"""Retained GP run identity validation before request construction."""

from typing import Any, cast


def validate_gp_root_run_identity(
    raw: dict[str, Any],
    *,
    seed: Any,
    start: Any,
    count: Any,
    sampling: Any,
    construction: Any,
) -> None:
    if raw.get("version") in ("gp-root-calibration-v1", "gp-root-calibration-query-v1"):
        original = raw.get("original")
        if type(original) is not dict:
            raise ValueError("GP calibration requires retained original source identity")
        raw = cast(dict[str, Any], original)
    source = raw.get("source")
    if type(source) is not dict:
        raise ValueError("GP root run requires exactly one retained source identity")
    source = cast(dict[str, Any], source)
    if (
        any(type(v) is not int for v in (seed, start, count))
        or count != 1
        or any(
            type(source.get(k)) is not int or source[k] != value
            for k, value in (("seed", seed), ("task_id", start))
        )
        or sampling is not None
        or construction is not None
    ):
        raise ValueError("GP root run requires exactly one retained source identity")
