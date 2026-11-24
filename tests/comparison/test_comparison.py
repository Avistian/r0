"""Independent fixtures exercise the public comparison boundary."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.comparison import (
    attempt,
    cache_identity,
    check_invariance,
    compare,
    predict,
    prepare,
    read_attempt,
    row_identity,
    score,
    tabpfn_status,
)


def fixture(tmp_path: Path, task: str = "binary", count: int = 36) -> Path:
    classes = [] if task == "regression" else ["b", "a"] if task == "binary" else ["c", "a", "b"]
    data = {
        "evidence": "AUTHORED_FIXTURE",
        "dataset": "authored-test",
        "task": task,
        "classes": classes,
        "features": [{"name": "x", "kind": "numeric"}, {"name": "group", "kind": "categorical"}],
        "rows": [
            {
                "values": [i, None if i % 7 == 0 else str(i % 5)],
                "target": i * 2.0 if not classes else classes[i % len(classes)],
            }
            for i in range(count)
        ],
    }
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(data))
    return source


def view() -> dict[str, Any]:
    return {
        "version": 1,
        "evidence": "AUTHORED_FIXTURE",
        "task": "binary",
        "classes": ["b", "a"],
        "features": [
            {"name": "x", "kind": "numeric"},
            {"name": "empty", "kind": "numeric"},
            {"name": "group", "kind": "categorical"},
        ],
        "context_ids": ["c0", "c1", "c2", "c3"],
        "query_ids": ["q0", "q1"],
        "context_values": [
            [1, None, "known"],
            [3, None, None],
            [None, None, "known"],
            [5, None, "other"],
        ],
        "query_values": [[100000, 900, "unseen"], [None, None, None]],
        "context_targets": ["b", "a", "b", "a"],
        "support": 4,
        "limits": {},
    }


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_prepare_same_rows_for_all_predictors(tmp_path: Path, task: str) -> None:
    source = fixture(tmp_path, task)
    a = prepare(source, tmp_path / "a")
    b = prepare(source, tmp_path / "b")
    assert a == b
    assert len(a["query_ids"]) == 4
    for draw in a["draws"]:
        current = json.loads((tmp_path / "a" / draw["input"]).read_text())
        assert current["support"] == 8
        assert set(current["context_ids"]).isdisjoint(a["query_ids"])
        assert "targets" not in current
        assert "query_targets" not in current
        assert current["query_ids"] == a["query_ids"]
        if task != "regression":
            assert set(current["context_targets"]) == set(a["classes"])


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_real_dimensions(tmp_path: Path, task: str) -> None:
    manifest = prepare(
        fixture(tmp_path, task, 600), tmp_path / "prepared", query_count=256, context_count=256
    )
    assert len(manifest["query_ids"]) == 256
    assert all(
        len(d["context_ids"]) == len(set(d["context_ids"])) == 256 for d in manifest["draws"]
    )


@pytest.mark.parametrize(
    "name", ["simple", "linear", "boosted_tree", "learner_fixture", "tabpfn_fixture"]
)
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_predictors_context_only_and_invariant(name: str, task: str) -> None:
    input_view = view()
    input_view["task"] = task
    if name == "boosted_tree":
        # sklearn 1.9.1 cannot bin an entirely missing numeric context column.
        input_view["context_values"][0][1] = 0.0
    if task == "regression":
        input_view["classes"] = []
        input_view["context_targets"] = [10.0, 12.0, 14.0, 16.0]
    elif task == "multiclass":
        input_view["classes"] = ["b", "a", "c"]
        input_view["context_targets"] = ["b", "a", "c", "a"]
    calls: list[tuple[str, tuple[str, ...]]] = []
    prediction, evidence = predict(
        input_view, name, spy=lambda phase, ids: calls.append((phase, ids))
    )
    assert calls
    assert all(ids == tuple(input_view["context_ids"]) for _, ids in calls)
    assert evidence["fit_ids"] == input_view["context_ids"]
    if name in ("linear", "boosted_tree"):
        assert [p for p, _ in calls] == ["preprocessing", "estimator"]
        assert evidence["preprocessing"]["columns"][0]["median"] == 3
        assert evidence["preprocessing"]["columns"][1]["median"] == 0
        assert evidence["preprocessing"]["columns"][2]["categories"] == ["known", "other"]
    reversed_view = {
        **input_view,
        "query_ids": input_view["query_ids"][::-1],
        "query_values": input_view["query_values"][::-1],
    }
    reversed_prediction, _ = predict(reversed_view, name)
    chunks = [
        predict(
            {
                **input_view,
                "query_ids": [input_view["query_ids"][i]],
                "query_values": [input_view["query_values"][i]],
            },
            name,
        )[0]
        for i in range(2)
    ]
    check_invariance(prediction, reversed_prediction[::-1], np.concatenate(chunks))


def test_hand_computed_metrics_and_declared_ties() -> None:
    classification = score(
        "binary",
        ["yes", "no"],
        ["q1", "q2"],
        ["yes", "no"],
        ["q2", "q1"],
        np.array([[0.25, 0.75], [0.5, 0.5]]),
    )
    assert classification == {"brier": 0.3125, "accuracy": 1.0}
    regression = score(
        "regression", [], ["q1", "q2"], [1.0, 5.0], ["q1", "q2"], np.array([2.0, 3.0])
    )
    assert regression["mae"] == 1.5
    assert regression["rmse"] == pytest.approx(np.sqrt(2.5), abs=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1.2, -0.2], [0.5, 0.5]]),
        np.array([[0.2, 0.2], [0.5, 0.5]]),
        np.array([[np.nan, 0.0], [0.5, 0.5]]),
        np.ones((2, 3)),
    ],
)
def test_invalid_probabilities_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="invalid|nonfinite"):
        score("binary", ["b", "a"], ["q0", "q1"], ["b", "a"], ["q0", "q1"], bad)


@pytest.mark.parametrize("ids", [["q0", "q0"], ["q0"], ["q0", "extra"]])
def test_invalid_prediction_ids(ids: list[str]) -> None:
    with pytest.raises(ValueError, match="prediction rows"):
        score("regression", [], ["q0", "q1"], [1.0, 2.0], ids, np.ones(len(ids)))


def test_cache_identity_and_tampering(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare(fixture(tmp_path), prepared)
    path = tmp_path / "attempt"
    result = attempt(prepared, path, "simple", 1, diagnostic=True)
    assert result["status"] == "SUCCESS"
    identity = cache_identity(prepared, "simple", 1, diagnostic=True)
    assert read_attempt(path, identity) == result
    with pytest.raises(ValueError, match="identity mismatch"):
        read_attempt(path, {**identity, "draw": 2})
    with pytest.raises(FileExistsError):
        attempt(prepared, path, "simple", 1, diagnostic=True)
    (path / "output/predictions.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corruption"):
        read_attempt(path, identity)


def test_fail_closed_and_partial_results(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare(fixture(tmp_path), prepared)
    result = attempt(prepared, tmp_path / "attempt", "simple", 1)
    assert result["status"] == "FAILED"
    assert "cgroup" in result["reason"]
    assert (tmp_path / "attempt/integrity.json").exists()
    assert tabpfn_status()["status"] == "UNAVAILABLE"


def test_offline_smoke(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare(fixture(tmp_path), prepared)
    result = compare(prepared, tmp_path / "results", diagnostic=True, fixture_predictors=True)
    assert result["status"] == "INCOMPLETE"
    assert result["fixture_checks"] == "PASS"
    assert len(result["summaries"]) == 5
    assert result["resource_enforcement"] == "NOT_ENFORCED"


def test_identity_before_exclusions_and_real_gate(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    data = json.loads(source.read_text())
    data["rows"][1]["target"] = None
    source.write_text(json.dumps(data))
    result = prepare(source, tmp_path / "prepared")
    assert result["exclusions"][0]["id"] == row_identity(
        result["source_sha256"], result["parser"], 1
    )
    assert row_identity(result["source_sha256"], result["parser"], 2) != row_identity(
        result["source_sha256"], result["parser"], 1
    )
    data["evidence"] = "REAL"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="real execution blocked"):
        prepare(source, tmp_path / "real")


def test_no_query_targets_and_infeasible_support(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scoring targets"):
        predict({**view(), "query_targets": ["a", "b"]}, "linear")
    with pytest.raises(ValueError, match="infeasible"):
        prepare(fixture(tmp_path, count=8), tmp_path / "bad")
