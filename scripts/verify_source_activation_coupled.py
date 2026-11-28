"""Independent NumPy row-coupling oracle and malformed fitted-record defense."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sample = np.array([[-3.0, 2.0, 2.0], [0.0, 1.0, 1.0], [5.0, -1.0, -1.0], [2.0, 4.0, 4.0]])
query = np.array([[8.0, -3.0, -3.0], [0.5, 0.5, 0.5]])
results = []


def primitive(value, kind):
    if kind == "softmax":
        exp = np.exp(value - np.max(value, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    if kind == "tied_max":
        return (value == np.max(value, axis=1, keepdims=True)).astype(float)
    if kind == "argsort":
        return np.argsort(value, axis=1, stable=True).astype(float)
    order = np.argsort(value, axis=1, stable=True)
    return np.argsort(order, axis=1, stable=True).astype(float)


for kind in ("softmax", "tied_max", "argsort", "rank"):
    mean = np.mean(sample, axis=0)
    inverse = 1 / (np.std(sample, axis=0) + 1e-4)
    standardized = (sample - mean) * inverse
    indices = np.random.default_rng(94).integers(len(sample), size=sample.shape[1])
    shift = -standardized[indices, np.arange(sample.shape[1])]
    transformed = primitive(3 * (standardized + shift), kind)
    output_mean = np.mean(transformed, axis=0)
    output_inverse = 1 / (np.std(transformed, axis=0) + 1e-4)
    expected = (
        primitive(3 * ((query - mean) * inverse + shift), kind) - output_mean
    ) * output_inverse
    activation = priors.calibrate_source_activation(
        priors.CausalActivation(kind),
        sample,
        rng=np.random.default_rng(94),
        scale=3.0,
        calibration_id=f"coupled-{kind}",
    )
    actual = activation.apply(torch.tensor(query)).numpy()
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-11
    results.append({"kind": kind, "max_error": error, "record": asdict(activation)})
    print(kind, "PASS", error, flush=True)

controls = priors.LinearRecipe(
    rows=16, context_rows=8, features=1, signal_features=1, active_features=1
)
one = priors.calibrate_source_activation(
    priors.CausalActivation(),
    np.array([[-1.0], [1.0]]),
    rng=np.random.default_rng(0),
    scale=1.0,
    calibration_id="decode",
)
recipe = priors.CausalRecipe(
    task=controls,
    nodes=(priors.ActivatedCausalNode(output_activation=one), priors.CausalNode()),
    observed=((1, 0),),
    target=((0, 0),),
    version="causal-v2",
)
raw = json.loads(json.dumps(asdict(recipe)))
for key, invalid in (
    ("calibration_digest", "bad"),
    ("input_mean", [0.0, 1.0]),
    ("input_inverse_std", [-1.0]),
    ("shift_rows", [2]),
    ("calibration_version", "other"),
    ("scale", 11.0),
):
    changed = copy.deepcopy(raw)
    changed["nodes"][0]["output_activation"][key] = invalid
    try:
        priors.recipe_from_dict(changed)
    except ValueError as error:
        results.append({"invalid_field": key, "rejection": str(error)})
    else:
        raise AssertionError(f"accepted malformed calibrated recipe: {key}")
    print("malformed", key, "PASS", flush=True)
try:
    priors.CausalActivationPrior(choices=(one,))
except ValueError as error:
    results.append({"legacy_prior_rejection": str(error)})
    print("prior-constructor", type(error).__name__, str(error), flush=True)
else:
    raise AssertionError("legacy prior accepted a fitted wrapper")
(args.output / "coupled.json").write_text(json.dumps(results, indent=2) + "\n")

for task_type, classes, width in (("binary", 2, 1), ("multiclass", 3, 3)):
    fitted = priors.calibrate_source_activation(
        priors.CausalActivation(),
        np.array([[-1.0] * width, [1.0] * width]),
        rng=np.random.default_rng(28),
        scale=2.0,
        calibration_id=f"adapter-{task_type}",
    )
    controls = priors.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        classes=classes,
    )
    bias = tuple(float(i + 1) for i in range(width))
    recipe = priors.CausalRecipe(
        task=controls,
        nodes=(
            priors.ActivatedCausalNode(
                bias=bias, noise_scale=(0.0,) * width, output_activation=fitted
            ),
            priors.CausalNode(),
        ),
        observed=((1, 0),),
        target=tuple((0, i) for i in range(width)),
        version="causal-v2",
    )
    raw = json.dumps(asdict(recipe), indent=2)
    (args.output / f"{task_type}-recipe.json").write_text(raw + "\n")
    restored = priors.recipe_from_dict(json.loads(raw))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=0)
    expected = np.array(bias) * 2 / 2.00010001
    assert np.max(np.abs(task.scores.numpy() - expected)) < 1e-12
    trace = priors.explain_generation(task, [0, 15])
    (args.output / f"{task_type}-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    assert trace["target_max_error"] == 0.0
    assert set(task.targets.tolist()) <= set(range(classes))
    result = {
        "task_type": task_type,
        "expected_scores": expected.tolist(),
        "scores": task.scores.tolist(),
        "targets": task.targets.tolist(),
        "probabilities": task.probabilities.tolist(),
        "trace_error": trace["target_max_error"],
    }
    results.append(result)
    print("calibrated-saved-adapter", task_type, "PASS", flush=True)
(args.output / "coupled.json").write_text(json.dumps(results, indent=2) + "\n")
