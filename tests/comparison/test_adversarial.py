"""Boundary cases, independent allocation oracle and separate backend contract."""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.comparison import (
    Unsupported,
    aggregate_datasets,
    comparison_gaps,
    predict,
    predict_adapter,
    prepare,
    run_process,
    score,
)


def test_allocation_matches_exhaustive_integer_oracle(tmp_path: Path) -> None:
    counts = [19, 9, 4]
    classes = ["c", "a", "b"]
    rows = [
        {"values": [f"{c}-{i}"], "target": c}
        for c, n in zip(classes, counts, strict=True)
        for i in range(n)
    ]
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "evidence": "AUTHORED_FIXTURE",
                "dataset": "allocation",
                "task": "multiclass",
                "classes": classes,
                "features": [{"name": "x", "kind": "categorical"}],
                "rows": rows,
            }
        )
    )
    prepare(path, tmp_path / "prepared", query_count=9, context_count=8)
    labels = json.loads((tmp_path / "prepared/scoring.json").read_text())["targets"]
    actual = [labels.count(c) for c in classes]
    feasible = [
        a
        for a in itertools.product(range(1, 9), repeat=3)
        if sum(a) == 9 and all(x < n for x, n in zip(a, counts, strict=True))
    ]

    def objective(allocation: Any) -> float:
        return sum((x - 9 * n / sum(counts)) ** 2 for x, n in zip(allocation, counts, strict=True))

    assert objective(actual) == min(objective(a) for a in feasible)


def _view(count: int = 4) -> dict[str, Any]:
    return {
        "version": 1,
        "evidence": "AUTHORED_FIXTURE",
        "task": "regression",
        "classes": [],
        "features": [{"name": "category", "kind": "categorical"}],
        "context_ids": [f"c{i}" for i in range(count)],
        "query_ids": ["q"],
        "context_values": [[str(i)] for i in range(count)],
        "query_values": [["unseen"]],
        "context_targets": list(range(count)),
        "support": count,
        "limits": {},
    }


def test_category_limit_is_not_silently_grouped() -> None:
    with pytest.raises(Unsupported, match="255"):
        predict(_view(256), "boosted_tree")


def test_tree_empty_numeric_column_retains_compatibility_failure() -> None:
    view = _view()
    view["features"] = [{"name": "empty", "kind": "numeric"}]
    view["context_values"] = [[None]] * 4
    view["query_values"] = [[1.0]]
    # Current pinned verification environment cannot bin an all-NaN column.
    # If the library is revised this test must be reviewed, not silently retuned.
    with pytest.raises(ValueError, match="window shape"):
        predict(view, "boosted_tree")


def test_backend_sees_only_context_labels() -> None:
    view = _view()
    captured: list[dict[str, Any]] = []

    class FixtureLearner:
        def fit(self, context: dict[str, Any]) -> None:
            captured.append(context)

        def predict(
            self, queries: dict[str, Any]
        ) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
            assert "targets" not in queries
            return queries["row_ids"], np.array([1.5])

    result = predict_adapter(
        view,
        FixtureLearner(),
        {
            "implementation": "authored-mean-v1",
            "environment": "fixture",
            "settings": {"mean": True},
            "checkpoint": "FIXTURE_NO_REAL_CHECKPOINT",
            "evidence": "FIXTURE_ONLY",
        },
    )
    assert captured[0]["row_ids"] == view["context_ids"]
    assert captured[0]["targets"] == view["context_targets"]
    assert result["query_ids"] == ["q"]


def test_gaps_negative_quality_zero_denominator_and_partitioning() -> None:
    negative = comparison_gaps("binary", [0.6, 0.6, 0.6], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5])
    assert negative["status"] == "COMPLETE"
    assert not negative["quality_target_met"]
    regression = comparison_gaps("regression", [2.0, 4.0, 8.0], [1.0, 2.0, 4.0], [2.0, 2.0, 2.0])
    assert regression["gap"]["draws"] == [0.5, 1.0, 2.0]
    assert (
        comparison_gaps("regression", [2.0, 4.0, 8.0], [1.0, 2.0, 4.0], [0.0, 2.0, 2.0])["status"]
        == "INCOMPLETE"
    )
    datasets = [
        {"dataset": name, "partition": partition, "task": task, "comparison": result}
        for name, partition, task, result in [
            ("a", "development", "binary", negative),
            ("b", "milestone", "binary", negative),
            ("c", "development", "regression", regression),
        ]
    ]
    assert len(aggregate_datasets(datasets)["groups"]) == 3
    with pytest.raises(ValueError, match="duplicate dataset"):
        aggregate_datasets(datasets + datasets)


def test_scalar_output_rejected_and_multiclass_oracle() -> None:
    with pytest.raises(ValueError, match="missing predictions"):
        score("regression", [], ["q"], [1.0], ["q"], np.array(1.0))
    result = score("multiclass", ["c", "a", "b"], ["q"], ["b"], ["q"], np.array([[0.2, 0.3, 0.5]]))
    assert result["brier"] == pytest.approx(0.38, abs=1e-12)
    assert result["accuracy"] == 1.0


def test_resource_preflight_never_falls_back(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-run"
    with pytest.raises(ValueError, match="enforcement unavailable"):
        run_process(["touch", str(marker)], cwd=tmp_path, cgroup_root=tmp_path)
    assert not marker.exists()
