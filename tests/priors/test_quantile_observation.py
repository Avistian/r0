"""Context fits remain frozen through measurement, encoding, masking and persistence."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def source() -> p.TrainingTask:
    task = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=4, features=2, signal_features=2, active_features=2),
        seed=30,
        task_id=7,
    )
    inputs = task.inputs.clone()
    inputs[:8, 0] = torch.tensor([0.0, 0.0, 2.0, 4.0, -100.0, 1.0, 3.0, 100.0])
    return replace(
        task, inputs=inputs, context_indices=torch.arange(4), query_indices=torch.arange(4, 16)
    )


def test_context_only_numeric_fit_composes_and_replays(tmp_path: Path) -> None:
    task = source()
    result = p.observe_missingness(
        task,
        p.MissingnessPlan("mcar", (0,), probability=0),
        seed=42,
        quantiles=(p.QuantileColumnSpec(0, "quantile-uniform"),),
        measurements=(p.MeasurementStage("affine", (0,), (("scale", 2.0), ("shift", 1.0))),),
    )
    expected = np.array([0.25, 0.25, 0.625, 0.875, 0.0, 0.4375, 0.75, 1.0]) * 2 + 1
    np.testing.assert_array_equal(result.inputs[:8, 0].numpy(), expected)
    receipt = json.loads(result.recipe_json)["observation"]
    assert receipt["version"] == "r0-missing-observation-v4"
    assert receipt["quantile"]["calibration_rows"] == [0, 1, 2, 3]
    assert receipt["quantile"]["maps"][0]["knots"] == [0.0, 2.0, 4.0]
    path = tmp_path / "quantile.zip"
    g.save_group(path, [(7, result)], max_bytes=8000000)
    saved = next(g.load_group(path))
    torch.testing.assert_close(
        p.restore_observation_source(saved).inputs, task.inputs, rtol=0, atol=0
    )
    altered = task.inputs.clone()
    altered[4:, 0] *= 10
    changed = p.observe_missingness(
        replace(task, inputs=altered),
        p.MissingnessPlan("mcar", (0,), probability=0.3),
        seed=42,
        quantiles=(p.QuantileColumnSpec(0, "quantile-uniform"),),
    )
    assert json.loads(changed.recipe_json)["observation"]["quantile"] == receipt["quantile"]
    assert result.observation_truth is not None
    bad_raw = result.observation_truth.raw_inputs.clone()
    bad_raw[0, 0] = -9
    with pytest.raises(ValueError, match="quantile receipt"):
        replace(result, observation_truth=replace(result.observation_truth, raw_inputs=bad_raw))


def test_context_fit_rejects_discrete_columns_duplicate_rows_and_budgets() -> None:
    task = source()
    quantiles = (p.QuantileColumnSpec(0),)
    plan = p.MissingnessPlan("mcar", (0,), probability=0.3)
    with pytest.raises(ValueError, match="continuous"):
        p.observe_missingness(
            replace(
                task,
                inputs=torch.zeros_like(task.inputs),
                feature_schema=(p.TrainingFeature("count"),) * 2,
            ),
            plan,
            seed=42,
            quantiles=quantiles,
        )
    with pytest.raises(ValueError, match="context row"):
        p.observe_missingness(
            replace(task, context_indices=torch.tensor([0, 0])), plan, seed=42, quantiles=quantiles
        )
    with pytest.raises(ValueError, match="resource"):
        p.observe_missingness(
            task, plan, limits=replace(p.CausalLimits(), parameters=2), seed=42, quantiles=quantiles
        )


def test_guarded_quantile_encoding_mask_archive_resume(tmp_path: Path) -> None:
    request = g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=30,
        count=2,
        max_seconds=30,
        max_bytes=8000000,
        observation=g.ObservationRequest(
            p.MissingnessPlan("mcar", (0,), probability=0.3),
            quantiles=(p.QuantileColumnSpec(0, "rank"),),
            encodings=(
                p.ColumnEncoding(0, "binary", cutpoints=(0.5,), vocabulary=("low", "high")),
            ),
        ),
    )
    request = g.request_from_dict(json.loads(json.dumps(asdict(request))))
    first = g.run_guarded(request, tmp_path)
    assert first.returncode == 0, first.stderr
    assert first.summary is not None
    assert first.summary["completed"] == 2
    for archive in tmp_path.rglob("data-*.zip"):
        for saved in g.load_group(archive):
            p.validate_missing_observation(saved)
            assert saved.feature_schema[0].kind == "binary"
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
