"""Fresh per-task tree draws retain replayable, request-bound archive receipts."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, load_group, run_generation
from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.generation.lib.requests import record_matches, selected_recipe
from r0.modules.priors import LinearRecipe, TreeExtensionRequest, sampled_tree_extension_from_dict


def test_tree_construction_archive_and_resume(tmp_path: Path) -> None:
    task = LinearRecipe(rows=16, context_rows=10, features=2, signal_features=2, active_features=2)
    controls = {
        "topology": "tree-extension",
        "request": asdict(
            TreeExtensionRequest(
                task,
                min_depth=1,
                max_depth=1,
            )
        ),
    }
    request = RunRequest(
        task,
        seed=29,
        count=2,
        max_seconds=30,
        max_bytes=4_000_000,
        construction=GraphConstruction(json.dumps(controls), seed=42),
    )
    assert selected_recipe(request, 0) != selected_recipe(request, 1)
    result = run_generation(request, tmp_path)
    assert result.completed == 2
    saved = [t for p in tmp_path.rglob("data-*.zip") for t in load_group(p)]
    assert len(saved) == 2
    for task in saved:
        record = json.loads(task.recipe_json)
        identity = record["task_id"]
        receipt = sampled_tree_extension_from_dict(record["construction"]["sampled"])
        assert receipt.recipe == selected_recipe(request, identity)
        assert record_matches(record, request, identity)
        altered = deepcopy(record)
        altered["construction"]["construction_seed"] += 1
        assert not record_matches(altered, request, identity)
        altered = deepcopy(record)
        altered["construction"]["task_id"] = float(identity)
        assert not record_matches(altered, request, identity)
        assert not record_matches(record, request, 1 - identity)
    resumed = run_generation(request, tmp_path)
    assert resumed.completed == 2
    assert resumed.attempted == 0


def test_guarded_tree_example_round_trip(tmp_path: Path) -> None:
    from r0.modules.generation import load_request, request_from_dict, run_guarded

    request = load_request(Path("configs/generation/per-task-tree-extension-cpu.json"))
    assert request_from_dict(json.loads(json.dumps(asdict(request)))) == request
    first = run_guarded(request, tmp_path, workers=2)
    assert first.returncode == 0, first.stderr
    assert first.summary is not None
    assert first.summary["completed"] == 3
    resumed = run_guarded(request, tmp_path, workers=1)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
