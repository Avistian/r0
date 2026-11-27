"""Explicit links between retained sampling decisions and causal recipe fields."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class PreferenceBinding:
    kind: str
    draw: int
    path: tuple[str | int, ...]
    operation: str = "value"
    value_index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("numerical", "categorical") or self.operation not in (
            "value",
            "length",
        ):
            raise ValueError("unsupported preference binding kind or operation")
        if type(self.draw) is not int or self.draw < 0 or self.value_index != 0:
            raise ValueError(
                "matrix preference bindings require a nonnegative draw and scalar index"
            )
        if type(self.value_index) is not int or (
            type(self.path) is not tuple
            or not 3 <= len(self.path) <= 32
            or self.path[0] != "nodes"
            or any(
                not (
                    (type(v) is int and v >= 0)
                    or (type(v) is str and v.isidentifier() and not v.startswith("_"))
                )
                for v in self.path
            )
        ):
            raise ValueError("preference binding path must name public node record fields")

    @property
    def parameter_count(self) -> int:
        return 2 + sum(type(v) is int for v in self.path)

    def read(self, recipe: Any) -> Any:
        value: Any = recipe
        for key in self.path:
            if type(key) is int:
                if type(value) is not tuple:
                    raise ValueError("preference binding path requires a tuple")
                items = cast(tuple[Any, ...], value)
                if key >= len(items):
                    raise ValueError("preference binding index is absent from the recipe")
                value = items[key]
            elif is_dataclass(value) and key in {f.name for f in fields(value)}:
                value = getattr(value, key)
            else:
                raise ValueError("preference binding path is absent from the recipe")
        if self.operation == "length":
            if type(value) is not tuple:
                raise ValueError("preference length binding requires a tuple")
            return len(cast(tuple[Any, ...], value))
        return value
