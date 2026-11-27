"""Write-once paired causal archives; a final manifest binds both group receipts."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.generation.lib.groups import save_group, verify_group, write_record
from r0.modules.generation.lib.loading import load_group
from r0.modules.priors import CausalLimits, TaskPair, TrainingTask, compare_causal_pair


def _same_card(actual: Any, expected: dict[str, Any]) -> bool:
    if expected.get("version") == "recursive-population-graph-pair-v1":
        # Preserve legacy policies; the new population record owns exact primitive custody.
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False
        )
    return actual == expected


def save_causal_pair(
    path: Path, pair: TaskPair, *, max_bytes: int, limits: CausalLimits | None = None
) -> None:
    card = compare_causal_pair(pair.before, pair.after, limits=limits)
    if not _same_card(json.loads(pair.card_json), card):
        raise ValueError("causal pair card differs from replay")
    path.mkdir(parents=True, exist_ok=False)
    task_id = json.loads(pair.before.recipe_json)["task_id"]
    before = save_group(path / "before.zip", [(task_id, pair.before)], max_bytes=max_bytes)
    after = save_group(path / "after.zip", [(task_id, pair.after)], max_bytes=max_bytes)
    write_record(
        path / "pair.json",
        {
            "version": "causal-pair-archive-v1",
            "before": asdict(before),
            "after": asdict(after),
            "card": card,
        },
    )


def load_causal_pair(path: Path, *, limits: CausalLimits | None = None) -> TaskPair:
    try:
        manifest = json.loads((path / "pair.json").read_bytes())
        if (
            set(manifest) != {"version", "before", "after", "card"}
            or manifest["version"] != "causal-pair-archive-v1"
        ):
            raise ValueError("invalid causal pair manifest")
        tasks: list[TrainingTask] = []
        for role in ("before", "after"):
            receipt = verify_group(path / f"{role}.zip")
            if (
                json.loads(json.dumps(asdict(receipt))) != manifest[role]
                or len(receipt.task_ids) != 1
            ):
                raise ValueError("causal pair group differs from final manifest")
            tasks.append(next(load_group(path / f"{role}.zip")))
        card = compare_causal_pair(tasks[0], tasks[1], limits=limits)
        if not _same_card(manifest["card"], card):
            raise ValueError("causal pair evidence differs from retained replay")
        return TaskPair(tasks[0], tasks[1], json.dumps(card, sort_keys=True))
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("incomplete or malformed causal pair archive") from error
