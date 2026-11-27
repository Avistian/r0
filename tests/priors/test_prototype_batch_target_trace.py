"""Original-row target dependencies reproduce noise and categorical inverse CDF."""

import hashlib
import json
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from r0.modules import priors as p


def emitted(kind: Literal["regression", "binary", "multiclass"], *, zero: bool = False):
    width = 3 if kind == "multiclass" else 1
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=kind,
        classes=0 if kind == "regression" else 2 if kind == "binary" else 3,
        noise_ratio=1.5 if kind == "regression" else 0.0,
        separation=1.0 if kind == "regression" else 0.7,
        class_offsets=() if kind == "regression" else tuple(0.3 * i - 0.2 for i in range(width)),
    )
    network = p.FrozenNetworkFunction(
        1,
        (
            p.NetworkBlock(
                (0,),
                (
                    p.NetworkAffine(
                        tuple((0.0 if zero else float(i + 1),) for i in range(width)),
                        (0.0,) * width,
                    ),
                ),
            ),
        ),
    )
    return p.generate_task(
        p.PrototypeBatchNetworkRecipe(p.NetworkRecipe(task, network)), seed=5, task_id=7
    )


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("zero", [False, True])
def test_batch_target_trace_dependencies_and_literal_arithmetic(
    kind: Literal["regression", "binary", "multiclass"], zero: bool
):
    task = emitted(kind, zero=zero)
    rows = [15, 0, 15, 3]
    graph = p.explain_generation(task, rows)
    nodes = {node["id"]: node for node in graph["nodes"]}
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])

    def values(key: str):
        return np.asarray(nodes[key]["values"], dtype=np.float64)

    signal = values("signal")
    stream = "noise" if kind == "regression" else "labels"
    identity = json.dumps(("functional-causal-v1", 5, 7, (stream,)), separators=(",", ":"))
    rng = np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
    raw = rng.normal(size=16) if kind == "regression" else rng.uniform(size=16)
    np.testing.assert_array_equal(
        values("noise-draw" if kind == "regression" else "labels")[:, 0], raw[rows]
    )
    if kind == "regression":
        assert nodes["noise"]["parents"] == ["noise-draw"]
        assert nodes["target"]["parents"] == ["signal", "noise"]
        np.testing.assert_array_equal(values("noise"), 1.5 * values("noise-draw"))
        expected = signal[:, 0] + values("noise")[:, 0]
    else:
        offsets = np.arange(signal.shape[1], dtype=np.float64) * 0.3 - 0.2
        logits = 0.7 * signal + offsets
        np.testing.assert_allclose(values("logits"), logits, rtol=0, atol=1e-15)
        if kind == "binary":
            positive = 1 / (1 + np.exp(-logits[:, 0]))
            probs = np.stack((1 - positive, positive), axis=1)
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(values("probabilities"), probs, rtol=0, atol=1e-15)
        assert nodes["target"]["parents"] == ["probabilities", "labels"]
        expected = (values("labels") >= probs.cumsum(axis=1)[:, :-1]).sum(axis=1)
    np.testing.assert_array_equal(values("target")[:, 0], expected)
    np.testing.assert_array_equal(values("targets")[:, 0], task.targets[rows].numpy())
    assert nodes["targets"]["parents"] == ["target"]
    assert graph["target_max_error"] == 0
    assert graph["rows"] == rows


def test_batch_trace_rejects_altered_probabilities_or_classes():
    task = emitted("multiclass")
    assert task.probabilities is not None
    for bad in (
        replace(task, probabilities=task.probabilities.roll(1, 1)),
        replace(task, probabilities=None),
        replace(task, classes=("wrong",)),
    ):
        with pytest.raises(ValueError, match="replay"):
            p.explain_generation(bad, [0, 1])
