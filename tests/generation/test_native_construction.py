"""Native construction binds transport row controls before guarded generation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession
from r0.modules.generation import GraphConstruction, RunRequest
from r0.modules.priors import LinearRecipe, recipe_from_dict


def test_native_transport_binds_preconstruction_geometry(tmp_path: Path) -> None:
    controls = ExplorerSession(tmp_path).request("construction_examples", {})["examples"][
        "Native mixed graph (sampled domains)"
    ]
    controls.update(rows=16, context_rows=7, split=31)
    construction = GraphConstruction(json.dumps(controls), 12)
    task = recipe_from_dict(construction.task_record())
    assert isinstance(task, LinearRecipe)
    assert (task.rows, task.context_rows, task.features, task.split) == (16, 7, 3, 31)
    RunRequest(
        recipe=task, construction=construction, seed=19, count=1, max_seconds=60, max_bytes=2**20
    )
    with pytest.raises(ValueError, match="task controls differ"):
        RunRequest(
            recipe=replace(task, rows=17),
            construction=construction,
            seed=19,
            count=1,
            max_seconds=60,
            max_bytes=2**20,
        )
