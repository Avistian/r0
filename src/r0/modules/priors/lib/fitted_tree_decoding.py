"""Bounded structural decoding; replay never fits or imports sklearn."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.fitted_tree_records import (
    FittedSplitTable,
    FittedTreeOutput,
    FrozenFittedTreeFunction,
)


def fitted_object(value: Any, kind: type[Any]) -> dict[str, Any]:
    if type(value) is not dict or set(cast(dict[str, Any], value)) - {f.name for f in fields(kind)}:
        raise ValueError("invalid fitted-tree record fields")
    return dict(cast(dict[str, Any], value))


def fitted_array(value: Any, cap: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) > cap:
        raise ValueError("invalid or oversized fitted-tree array")
    return tuple(value)


def decode_fitted_tree(data: dict[str, Any]) -> FrozenFittedTreeFunction:
    raw = fitted_object(data, FrozenFittedTreeFunction)
    outputs: list[FittedTreeOutput] = []
    count = nodes = 0
    for output in fitted_array(raw.get("outputs"), 16):
        values = fitted_object(output, FittedTreeOutput)
        tables: list[FittedSplitTable] = []
        for table in fitted_array(values.get("trees"), 3200):
            count += 1
            entry = fitted_object(table, FittedSplitTable)
            for key in ("features", "thresholds", "left", "right"):
                entry[key] = fitted_array(entry.get(key), 511)
            nodes += len(entry["features"])
            if count > 4096 or nodes > 131072:
                raise ValueError("fitted-tree decoding node/tree budget exceeded")
            entry["values"] = tuple(
                fitted_array(row, 100) for row in fitted_array(entry.get("values"), 511)
            )
            tables.append(FittedSplitTable(**entry))
        values["trees"] = tuple(tables)
        values["classes"] = fitted_array(values.get("classes", ()), 100)
        values["initial"] = fitted_array(values.get("initial", (0.0,)), 100)
        outputs.append(FittedTreeOutput(**values))
    raw["outputs"] = tuple(outputs)
    return FrozenFittedTreeFunction(**raw)
