"""Inspection summarizes the full task while retaining bounded, stable row views."""

import csv
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from r0.modules.explorer import inspect_rows, inspect_task, replay_bundle
from r0.modules.priors import LinearRecipe, generate_task


def test_full_task_summaries_and_bounded_rows_are_separate() -> None:
    task = generate_task(
        LinearRecipe(rows=513, context_rows=300, noise_ratio=0.7), seed=27, task_id=5
    )
    before = task.inputs.clone(), task.targets.clone(), task.scores.clone()
    view = inspect_task(task, sample_rows=17)
    assert view["total_rows"] == 513
    assert view["shown_rows"] == 17
    assert view["summary_scope"] == "all_rows"
    assert len(view["rows"]) == 17
    assert view["features"][0]["summary"]["mean"] == pytest.approx(
        np.mean(task.inputs.numpy()[:, 0]), abs=1e-15
    )
    assert sum(view["features"][0]["summary"]["histogram"]["counts"]) == 513
    assert view["split_counts"] == {"context": 300, "query": 213}
    assert view == inspect_task(task, sample_rows=17)
    chosen = inspect_rows(task, [0, 512])
    assert [row["index"] for row in chosen] == [0, 512]
    assert chosen[0]["inputs"] == np.asarray(task.inputs[0].numpy(), dtype=np.float64).tolist()
    assert chosen[0]["noise"] == float(task.targets[0] - task.scores[0, 0])
    for expected, actual in zip(before, (task.inputs, task.targets, task.scores), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    json.dumps(view, allow_nan=False)


def test_class_summaries_include_absent_classes_and_expected_counts() -> None:
    task = generate_task(
        LinearRecipe(
            rows=16,
            context_rows=8,
            task_type="multiclass",
            classes=3,
            separation=0,
            class_offsets=(0, -100, -100),
        ),
        seed=27,
        task_id=0,
    )
    view = inspect_task(task, sample_rows=4)
    assert view["classification"]["observed_counts"] == [16, 0, 0]
    assert task.probabilities is not None
    np.testing.assert_allclose(
        view["classification"]["expected_counts"], task.probabilities.sum(0).numpy()
    )
    assert len(view["rows"][0]["probabilities"]) == 3


def test_replay_bundle_preserves_complete_arrays_and_separates_predictor_truth() -> None:
    task = generate_task(LinearRecipe(rows=32, context_rows=20), seed=27, task_id=7)
    with zipfile.ZipFile(io.BytesIO(replay_bundle(task))) as bundle:
        assert set(bundle.namelist()) == {
            "recipe.json",
            "identity.json",
            "mechanism.json",
            "predictor.npz",
            "truth.npz",
            "predictor.csv",
            "truth.csv",
        }
        assert json.loads(bundle.read("identity.json"))["source_task_id"] == task.source_task_id
        with np.load(io.BytesIO(bundle.read("predictor.npz")), allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "context_x",
                "context_y",
                "query_x",
                "context_rows",
                "query_rows",
            }
            np.testing.assert_array_equal(
                arrays["context_x"], task.inputs[task.context_indices].numpy()
            )
        with np.load(io.BytesIO(bundle.read("truth.npz")), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["targets"], task.targets.numpy())
            np.testing.assert_array_equal(arrays["scores"], task.scores.numpy())
        header, *rows = bundle.read("predictor.csv").decode().splitlines()
        assert "score" not in header
        assert "probability" not in header
        assert len(rows) == 32
        table = list(csv.DictReader(io.StringIO(bundle.read("predictor.csv").decode())))
        assert all(row["context_target"] == "" for row in table if row["role"] == "query")
        assert sum(row["role"] == "query" for row in table) == 12
        assert len(bundle.read("truth.csv").decode().splitlines()) == 33


def test_inspection_handles_large_finite_scales_and_reports_export_limits() -> None:
    from r0.modules.priors import GroupedLinearRecipe, InputDistribution

    task = generate_task(
        GroupedLinearRecipe(inputs=InputDistribution(scale=1e200), calibration_rows=256),
        seed=27,
        task_id=2,
    )
    view = inspect_task(task, sample_rows=4)
    assert view["features"][0]["summary"]["std"] is not None
    assert view["features"][0]["summary"]["std"] > 1e199
    json.dumps(view, allow_nan=False)
    with pytest.raises(ValueError, match="sample_rows"):
        inspect_task(task, sample_rows=0)
    with pytest.raises(ValueError, match="indices"):
        inspect_rows(task, [-1])
    larger = generate_task(LinearRecipe(rows=1024, context_rows=512), seed=27, task_id=0)
    with pytest.raises(ValueError, match="output-storage limit"):
        replay_bundle(larger, max_bytes=65536)
