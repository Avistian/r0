"""Wrong GP producer envelopes reject before covariance reconstruction."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.generation import RunRequest, run_generation


@pytest.mark.parametrize(
    "change", ["target_family", "noise", "source_identity", "sampling", "version"]
)
def test_wrong_gp_producer_metadata_rejects_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
):
    recipe = p.GPColumnRecipe(
        p.GPColumnRequest(rows=16),
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
    )
    altered = recipe
    if change == "target_family":
        altered = replace(recipe, source=replace(recipe.request, target_family="quadratic"))
    elif change == "noise":
        altered = replace(recipe, task=replace(recipe.task, noise_ratio=0.5))
    supplied = p.generate_task(altered, seed=11, task_id=17)
    raw = json.loads(supplied.recipe_json)
    if change == "source_identity":
        raw["recipe"]["source"]["task_id"] = 18
    elif change == "sampling":
        raw["sampling"] = {"family": "unrequested"}
    elif change == "version":
        raw["recipe"]["version"] = "unknown-gp"
    supplied = replace(supplied, recipe_json=json.dumps(raw))

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("wrong producer metadata replayed covariance")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    result = run_generation(
        RunRequest(recipe, 11, 1, 30, 10_000_000, start=17),
        tmp_path,
        produce=lambda _: supplied,
    )
    assert result.failed == 1
    assert result.completed == 0
