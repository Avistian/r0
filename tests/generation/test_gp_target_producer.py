"""Fixed GP environment producers cannot relabel side, source, task or identity."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.mark.parametrize("change", ["side", "task", "source", "seed_type", "sampling"])
def test_gp_environment_wrong_producer_rejects_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    pop = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=11, task_id=17)
    recipe = p.GPColumnRecipe(
        pop,
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
    )
    pair = p.generate_gp_column_pair(recipe, pop.target)
    envelope = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    raw = json.loads(pair.after.recipe_json)
    if change == "side":
        raw["recipe"]["side"] = "original"
    if change == "task":
        raw["recipe"]["task"]["noise_ratio"] = 0.5
    if change == "source":
        raw["recipe"]["comparison"]["original"]["task_id"] = 18
    if change == "seed_type":
        raw["seed"] = 11.0
    if change == "sampling":
        raw["sampling"] = {"family": "unrequested"}
    supplied = replace(pair.after, recipe_json=json.dumps(raw))
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(1)
        raise AssertionError("wrong GP environment producer replayed covariance")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    result = g.run_generation(
        g.RunRequest(envelope, 11, 1, 30, 10_000_000, start=17),
        tmp_path,
        produce=lambda _: supplied,
    )
    assert result.failed == 1
    assert result.completed == 0
    assert not calls
