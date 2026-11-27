"""Saved network targets retain their original stochastic adapter dependencies."""

import hashlib
import json
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors as p


def emitted(kind: str, view: bool | str) -> p.TrainingTask:
    task_type: Literal["regression", "binary", "multiclass"] = (
        "regression"
        if kind in ("deterministic", "noisy")
        else "binary"
        if kind == "binary"
        else "multiclass"
    )
    outputs = 3 if task_type == "multiclass" else 1
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes=3 if task_type == "multiclass" else 2 if task_type == "binary" else 0,
        noise_ratio=1.25 if kind == "noisy" else 0.0,
        separation=1.0 if task_type == "regression" else 0.7,
        class_offsets=()
        if task_type == "regression"
        else tuple(0.2 * i - 0.4 for i in range(outputs)),
    )
    function = p.FrozenNetworkFunction(
        2,
        (
            p.NetworkBlock(
                (0,),
                (
                    p.NetworkAffine(
                        tuple((float(i + 1), -0.5) for i in range(outputs)), (0.3,) * outputs
                    ),
                ),
            ),
        ),
    )
    base = p.NetworkRecipe(task, function)
    recipe = (
        p.NetworkInputView(base, replace(task, features=3, active_features=1), (1,))
        if view
        else base
    )
    if view == "matched":
        matching = p.fit_network_variance_match(
            base,
            before_inputs=(0, 1),
            after_inputs=(1,),
            rows=32,
            rng=np.random.default_rng(47),
        )
        recipe = p.MatchedNetworkInputView(
            base, replace(task, features=3, active_features=1), (1,), matching=matching
        )
    return p.generate_task(recipe, seed=2**54 + 1, task_id=2**54 + 3)


@pytest.mark.parametrize("kind", ["deterministic", "noisy", "binary", "multiclass"])
@pytest.mark.parametrize("view", [False, True, "matched"])
def test_literal_network_target_dependencies(kind: str, view: bool | str) -> None:
    task = emitted(kind, view)
    indices = [15, 0, 15, 3]
    graph = p.explain_generation(task, indices)
    nodes = {n["id"]: n for n in graph["nodes"]}
    stream = "noise" if kind in ("deterministic", "noisy") else "labels"
    text = json.dumps(
        ("functional-causal-v1", 2**54 + 1, 2**54 + 3, (stream,)), separators=(",", ":")
    )
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest(), 16))
    raw = rng.normal(size=16) if stream == "noise" else rng.uniform(size=16)

    def values(key: str):
        return np.asarray(nodes[key]["values"], dtype=np.float64)

    signal: NDArray[np.float64] = task.scores.numpy()[indices]
    if stream == "noise":
        np.testing.assert_array_equal(values("noise-draw")[:, 0], raw[indices])
        assert nodes["noise"]["parents"] == ["noise-draw"]
        assert nodes["target"]["parents"] == ["signal", "noise"]
        expected = signal[:, 0] + (1.25 if kind == "noisy" else 0.0) * raw[indices]
    else:
        logits = 0.7 * signal + np.arange(signal.shape[1], dtype=np.float64) * 0.2 - 0.4
        if kind == "binary":
            positive = 1 / (1 + np.exp(-logits[:, 0]))
            probs = np.column_stack((1 - positive, positive))
        else:
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = weights / weights.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(values("probabilities"), probs, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(values("labels")[:, 0], raw[indices])
        assert nodes["target"]["parents"] == ["probabilities", "labels"]
        expected = (raw[indices, None] >= probs.cumsum(axis=1)[:, :-1]).sum(axis=1)
    assert nodes["targets"]["parents"] == ["target"]
    np.testing.assert_array_equal(values("targets")[:, 0], expected)
    np.testing.assert_array_equal(values("targets")[:, 0], task.targets.numpy()[indices])
    assert graph["target_max_error"] == 0


@pytest.mark.parametrize("view", [False, True, "matched"])
@pytest.mark.parametrize(
    "field",
    [
        "task_type",
        "source_task_id",
        "episode_id",
        "classes",
        "feature_schema",
        "target_origin",
        "mechanism_json",
        "implementation",
        "scores",
        "target_dtype",
    ],
)
def test_network_trace_rejects_semantic_metadata(view: bool | str, field: str) -> None:
    task = emitted("multiclass", view)
    if field == "task_type":
        bad = replace(task, task_type="regression")
    elif field == "source_task_id":
        bad = replace(task, source_task_id="wrong")
    elif field == "episode_id":
        bad = replace(task, episode_id="wrong")
    elif field == "classes":
        bad = replace(task, classes=("a", "b", "c"))
    elif field == "feature_schema":
        bad = replace(task, feature_schema=(p.TrainingFeature(),) * task.inputs.shape[1])
    elif field == "target_origin":
        bad = replace(
            task,
            target_origin="native-variable-v1",
            feature_schema=(p.TrainingFeature(),) * task.inputs.shape[1],
        )
    elif field == "mechanism_json":
        bad = replace(task, mechanism_json="{}")
    elif field == "implementation":
        record = json.loads(task.recipe_json)
        record["execution"]["implementation"] = "wrong"
        bad = replace(task, recipe_json=json.dumps(record))
    elif field == "scores":
        bad = replace(task, scores=task.scores + 1)
    else:
        bad = replace(task, targets=task.targets.to(dtype=task.inputs.dtype))
    with pytest.raises(ValueError, match="replay|metadata|arrays"):
        p.explain_generation(bad, [15, 0, 15])


@pytest.mark.parametrize("view", [False, True, "matched"])
def test_network_trace_probability_and_split_custody(view: bool | str) -> None:
    task = emitted("multiclass", view)
    assert task.probabilities is not None
    for bad in (
        replace(task, probabilities=None),
        replace(task, probabilities=task.probabilities.roll(1, 1)),
        replace(task, context_indices=task.context_indices.flip(0)),
        replace(task, query_indices=task.query_indices.flip(0)),
    ):
        with pytest.raises(ValueError, match="replay"):
            p.explain_generation(bad, [15, 0, 15])


@pytest.mark.parametrize("view", [False, True, "matched"])
def test_network_trace_retains_historical_environment_and_formats(view: bool | str) -> None:
    task = emitted("noisy", view)
    record = json.loads(task.recipe_json)
    record["execution"].update(
        python="historical-runtime", numpy="historical-numpy", device="historical-device"
    )
    changed = replace(
        task,
        recipe_json=json.dumps(record),
        mechanism_json=json.dumps(json.loads(task.mechanism_json), indent=2),
    )
    assert p.explain_generation(changed, [15, 0, 15]) == p.explain_generation(task, [15, 0, 15])
