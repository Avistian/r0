"""Finite GP feature populations through the public generation interface."""

import math
from pathlib import Path
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.episodes import TaskType


def test_two_row_column_draw_has_literal_cholesky_values():
    draw = p.build_gp_columns(
        ((0.0,), (math.sqrt(2 * math.log(2)),)),
        ((2.0,), (0.0,)),
        kernels=(p.GPColumnKernel(length_scale=1.0),),
    )
    assert draw.values == ((2.0,), (1.0,))
    assert draw.factors[0][1] == pytest.approx((0.5, math.sqrt(3) / 2))


@pytest.mark.parametrize("family", ["linear", "quadratic", "neural"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_public_generation_uses_gp_features_and_independent_target_calibration(
    family: str, task_type: TaskType, classes: int
):
    import json
    from dataclasses import asdict, replace

    import torch

    task = p.LinearRecipe(
        rows=16,
        context_rows=10,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes=classes,
    )
    request = p.GPColumnRequest(
        rows=16, columns=2, target_family=family, outputs=3 if task_type == "multiclass" else 1
    )
    recipe = p.GPColumnRecipe(request, task)
    result = p.generate_task(recipe, seed=11, task_id=7)
    saved = p.recipe_from_dict(json.loads(result.recipe_json)["recipe"])
    replay = p.generate_task(saved, seed=11, task_id=7)
    assert torch.equal(result.inputs, replay.inputs)
    assert torch.equal(result.targets, replay.targets)
    assert json.loads(result.mechanism_json)["family"] == "gp_columns"
    population = p.realize_gp_columns(request, seed=11, task_id=7)
    split = p.generate_task(
        replace(recipe, task=replace(task, context_rows=7, split=9)), seed=11, task_id=7
    )
    assert torch.equal(result.scores, split.scores)
    assert population.calibration_inputs != population.draw.values
    assert p.gp_column_population_from_dict(asdict(population)) == population


def test_archive_and_inspection_keep_joint_axes_and_repeated_rows(tmp_path: Path):
    import torch

    from r0.modules import generation as g

    recipe = p.GPColumnRecipe(
        p.GPColumnRequest(rows=16),
        p.LinearRecipe(rows=16, context_rows=10, features=2, signal_features=2, active_features=2),
    )
    task = p.generate_task(recipe, seed=3, task_id=8)
    path = tmp_path / "gp.zip"
    g.save_group(path, [(8, task)], max_bytes=10_000_000)
    loaded = next(g.load_group(path))
    trace = p.explain_generation(loaded, [7, 2, 7])
    assert trace["shared_matrices"][0]["row_ids"] == list(range(16))
    assert trace["shared_matrices"][0]["column_row_ids"] == list(range(16))
    inputs = next(n for n in trace["nodes"] if n["id"] == "inputs")
    assert inputs["values"] == [[float(v) for v in task.inputs[i]] for i in (7, 2, 7)]
    assert torch.equal(loaded.inputs, task.inputs)


def test_matched_and_fresh_comparisons_keep_target_mechanisms_and_calibration():
    from dataclasses import asdict

    population = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=4, task_id=2)
    target = p.GPColumnTarget("linear", ((1.0,), (0.0,)), (0.0,))
    matched = p.compare_gp_columns(population, target)
    fresh = p.compare_gp_columns(population, target, mode="fresh", fresh_seed=19)
    assert matched.draw == population.draw
    assert fresh.draw.locations == population.draw.locations
    assert fresh.draw.values != population.draw.values
    assert matched.centers == fresh.centers
    assert matched.scales == fresh.scales
    assert matched.original.target == fresh.original.target == population.target
    assert p.gp_column_comparison_from_dict(asdict(fresh)) == fresh


def test_gp_population_rejects_singular_locations_and_nonfinite_kernel():
    with pytest.raises(ValueError, match="positive definite"):
        p.build_gp_columns(((0.0,), (0.0,)), ((1.0,), (2.0,)))
    with pytest.raises(ValueError, match="nonfinite"):
        p.build_gp_columns(
            ((0.0,), (1.0,)), ((1.0,), (2.0,)), kernels=(p.GPColumnKernel(amplitude=1e200),)
        )


def test_admission_rejects_before_rng_or_dense_allocation(monkeypatch: pytest.MonkeyPatch):
    from dataclasses import asdict, replace

    import numpy as np

    request = p.GPColumnRequest(rows=16)
    population = p.realize_gp_columns(request, seed=5, task_id=3)
    raw = asdict(population)
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        raise AssertionError("numerical execution before admission")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    tiny = p.CausalLimits(parameters=10, working_bytes=1024)
    with pytest.raises(ValueError, match="limit"):
        replace(request, limits=tiny)
    with pytest.raises(ValueError, match="limit"):
        p.gp_column_population_from_dict(raw, limits=tiny)
    assert calls == []


@pytest.mark.parametrize("field", ["seed", "population_id", "row_ids", "scores", "centers"])
def test_population_archive_rejects_forged_identity_or_values(field: str):
    from dataclasses import asdict

    population = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=5, task_id=3)
    raw = asdict(population)
    raw[field] = {
        "seed": 6,
        "population_id": "forged",
        "row_ids": tuple(reversed(range(16))),
        "scores": ((0.0,),) * 16,
        "centers": (999.0,),
    }[field]
    with pytest.raises(ValueError, match="replay"):
        p.gp_column_population_from_dict(raw)


def test_all_target_equations_have_literal_values():
    linear = p.GPColumnTarget("linear", ((2.0,), (-1.0,)), (3.0,))
    quadratic = p.GPColumnTarget(
        "quadratic", ((2.0,), (-1.0,)), (3.0,), (((1.0, 2.0), (0.0, -1.0)),)
    )
    neural = p.GPColumnTarget(
        "neural", ((2.0,),), (1.0,), hidden=((1.0,), (-1.0,)), hidden_bias=(0.0,)
    )
    assert p.evaluate_gp_column_target(linear, ((2.0, 3.0),)) == ((4.0,),)
    assert p.evaluate_gp_column_target(quadratic, ((2.0, 3.0),)) == ((11.0,),)
    assert p.evaluate_gp_column_target(neural, ((2.0, 2.0),)) == ((1.0,),)


def test_per_column_kernels_retain_separate_factors():
    draw = p.build_gp_columns(
        ((0.0,), (math.sqrt(2 * math.log(2)),)),
        ((2.0, 2.0), (0.0, 0.0)),
        kernels=(p.GPColumnKernel(1.0, 1.0), p.GPColumnKernel(1.0, 2.0)),
    )
    assert draw.values == ((2.0, 4.0), (1.0, 2.0))
    assert len(draw.factors) == 2


def test_comparison_decoder_admits_replacement_before_original_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    from dataclasses import asdict, replace

    import numpy as np

    original = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=9, task_id=2)
    target = p.GPColumnTarget("linear", ((1.0,), (0.0,)), (0.0,))
    raw = asdict(p.compare_gp_columns(original, target))
    limits = replace(
        original.request.limits,
        working_bytes=4 * original.request.admit()["estimated_working_bytes"] + 1024,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("original replay started before aggregate comparison rejection")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="aggregate"):
        p.gp_column_comparison_from_dict(raw, limits=limits)


def test_realized_population_rejects_boolean_identity_alias():
    original = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=1, task_id=2)
    recipe = p.GPColumnRecipe(
        original,
        p.LinearRecipe(rows=16, context_rows=10, features=2, signal_features=2, active_features=2),
    )
    with pytest.raises(ValueError, match="integer identity"):
        p.generate_task(recipe, seed=True, task_id=2)
