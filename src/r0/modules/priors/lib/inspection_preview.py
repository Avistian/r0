"""Bounded parameter previews preserve explicit full-list sizes."""

from typing import Any, cast


def preview_parameters(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: preview_parameters(item) for key, item in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        items = cast(list[Any], value)
        shown = [preview_parameters(item) for item in items[:32]]
        return shown if len(items) <= 32 else {"total_items": len(items), "shown": shown}
    return value
