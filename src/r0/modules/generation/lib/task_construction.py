"""Explicit per-task construction streams for qualified sampled graph/network paths."""

import hashlib
import json
from typing import Any, cast

from r0.modules.generation.lib.construction import GraphConstruction

_VERSION = "per-task-construction-v1"
_QUALIFIED = frozenset(
    {
        "explicit",
        "cauchy",
        "geometry",
        "attachment-geometry",
        "motif-geometry",
        "network",
        "network-hyperprior",
    }
)


def task_construction_source(construction: GraphConstruction) -> dict[str, Any]:
    data = json.loads(construction.controls_json)
    if set(data) != {"topology", "source", "version"} or (
        data["topology"] != "per-task"
        or data["version"] != _VERSION
        or type(data["source"]) is not dict
    ):
        raise ValueError("per-task construction requires its version and source controls")
    source = cast(dict[str, Any], data["source"])
    if type(source.get("topology")) is not str or source["topology"] not in _QUALIFIED:
        raise ValueError("source construction is not qualified for per-task resampling")
    return source


def per_task_construction(source: GraphConstruction) -> GraphConstruction:
    """Opt into new mechanism draws while retaining fixed-mode defaults elsewhere."""
    if type(source) is not GraphConstruction:
        raise ValueError("per-task sampling requires a graph construction")
    result = GraphConstruction(
        json.dumps(
            {
                "topology": "per-task",
                "version": _VERSION,
                "source": json.loads(source.controls_json),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        source.seed,
    )
    task_construction_source(result)
    return result


def resolve_task_construction(
    construction: GraphConstruction, *, episode_seed: int, task_id: int
) -> tuple[GraphConstruction, dict[str, Any]]:
    source = task_construction_source(construction)
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (episode_seed, task_id)):
        raise ValueError("per-task construction identities must be exact bounded integers")
    # An odd affine map permutes 63-bit IDs, so seeds cannot collide within one
    # master/run-seed pair. Downstream constructors split this seed into streams.
    key = json.dumps([_VERSION, construction.seed, episode_seed], separators=(",", ":")).encode()
    digest = hashlib.sha256(key).digest()
    mask = 2**63 - 1
    multiplier = (int.from_bytes(digest[:8], "little") & mask) | 1
    offset = int.from_bytes(digest[8:16], "little") & mask
    derived = (multiplier * task_id + offset) & mask
    document = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return GraphConstruction(document, derived), {
        "version": _VERSION,
        "construction_seed": construction.seed,
        "episode_seed": episode_seed,
        "task_id": task_id,
        "derived_seed": derived,
        "source_sha256": hashlib.sha256(document.encode()).hexdigest(),
    }
