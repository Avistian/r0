"""Source identity for complete construction archives with nested episode controls."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.linear import LinearRecipe


def construction_source_config(
    config: dict[str, Any], task: LinearRecipe, *, nested_graph: bool
) -> dict[str, object]:
    controls = {k: v for k, v in asdict(task).items() if k not in ("context_rows", "split")}
    graph = config["graph"] if nested_graph else config
    source: dict[str, object] = {
        **graph,
        "request": {**graph["request"], "task": controls},
        "runtime": {**graph["runtime"], "task": controls},
    }
    if nested_graph:
        return {**config, "request": {**config["request"], "task": controls}, "graph": source}
    return source
