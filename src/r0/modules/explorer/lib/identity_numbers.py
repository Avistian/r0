"""Exact browser identity transport shared by explorer workflows."""

from typing import Any, cast


def identity_numbers(value: Any, *, for_browser: bool) -> Any:
    """Keep 63-bit identities exact across JavaScript's 53-bit integer boundary."""
    if isinstance(value, list):
        return [identity_numbers(item, for_browser=for_browser) for item in cast(list[Any], value)]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in cast(dict[str, Any], value).items():
        if key in (
            "seed",
            "task_id",
            "split",
            "population",
            "reference_seed",
            "reference_task_id",
            "fresh_seed",
        ):
            if for_browser and type(item) is int and item > 2**53 - 1:
                item = str(item)
            elif not for_browser and isinstance(item, str) and item.isascii() and item.isdecimal():
                item = int(item)
        result[key] = identity_numbers(item, for_browser=for_browser)
    return result
