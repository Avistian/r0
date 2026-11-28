"""Independent Decimal population-moment and frozen-function audit via public APIs."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors

getcontext().prec = 1100
D = Decimal
EPS = D("0.0001")
parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
results = []


def stats(values):
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)).sqrt()
    return mean, 1 / (std + EPS)


def tanh(value):
    exp = (2 * value).exp()
    return (exp - 1) / (exp + 1)


def save():
    (args.output / "numerical.json").write_text(json.dumps(results, indent=2) + "\n")
    imports = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("r0.") and path and path.endswith(".py"):
            imports[name] = {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
    (args.output / "imports.json").write_text(json.dumps(imports, indent=2) + "\n")


maximum = np.finfo(np.float64).max
subnormal = np.nextafter(0.0, 1.0)
cases = {
    "original_reject": [1e16, 1e16 + 4],
    "negative_offset": [-1e16, -1e16 + 4],
    "adjacent_max": [np.nextafter(maximum, 0), maximum],
    "opposite_extremes": [-maximum, maximum],
    "asymmetric_extremes": [-maximum, maximum / 2, maximum],
    "constant_max": [maximum, maximum, maximum],
    "constant_zero": [0.0, 0.0],
    "constant_subnormal": [subnormal, subnormal],
    "subnormal_spread": [-subnormal, 0.0, subnormal],
    "ordinary": [-3.0, -1.0, 0.5, 2.0, 7.0],
}
for name, values in cases.items():
    sample = np.array(values, dtype="<f8")[:, None]
    (args.output / f"{name}.f64le").write_bytes(sample.tobytes())
    decimal_values = [D.from_float(float(v)) for v in values]
    mean, inverse = stats(decimal_values)
    activation = priors.calibrate_source_activation(
        priors.CausalActivation("tanh"),
        sample,
        rng=np.random.default_rng(0),
        scale=1.0,
        calibration_id=name,
    )
    # Stored means are binary64. Use that represented mean in the function oracle.
    represented_mean = D.from_float(float(mean))
    index = int(np.random.default_rng(0).integers(len(values), size=1)[0])
    shift = -(decimal_values[index] - represented_mean) * inverse
    outputs = [tanh((v - represented_mean) * inverse + shift) for v in decimal_values]
    output_mean, output_inverse = stats(outputs)
    expected = np.array([float((v - output_mean) * output_inverse) for v in outputs])
    actual = activation.apply(torch.tensor(sample)).numpy()[:, 0]
    relative_error = abs(activation.input_inverse_std[0] / float(inverse) - 1)
    max_error = float(np.max(np.abs(actual - expected)))
    result = {
        "name": name,
        "sample_hex": [float(v).hex() for v in values],
        "oracle_mean": str(mean),
        "oracle_inverse": str(inverse),
        "inverse_relative_error": relative_error,
        "function_absolute_error": max_error,
        "record": asdict(activation),
        "expected": expected.tolist(),
        "actual": actual.tolist(),
    }
    results.append(result)
    save()
    assert relative_error < 2e-13, result
    assert max_error < 2e-12, result
    assert math.isclose(activation.input_mean[0], float(mean), rel_tol=2e-15, abs_tol=5e-324)
    print(name, "PASS", relative_error, max_error, flush=True)

# Original rejecting query, saved authored generation and row trace, without refitting.
activation = priors.calibrate_source_activation(
    priors.CausalActivation("tanh"),
    np.array([[1e16], [1e16 + 4]]),
    rng=np.random.default_rng(0),
    scale=1.0,
    calibration_id="original-reject",
)
controls = priors.LinearRecipe(
    rows=16, context_rows=8, features=1, signal_features=1, active_features=1
)
recipe = priors.CausalRecipe(
    task=controls,
    nodes=(
        priors.ActivatedCausalNode(
            bias=(1e16 + 8,), noise_scale=(0.0,), output_activation=activation
        ),
        priors.CausalNode(),
    ),
    observed=((1, 0),),
    target=((0, 0),),
    version="causal-v2",
)
raw = json.dumps(asdict(recipe), indent=2)
(args.output / "repaired-recipe.json").write_text(raw + "\n")
restored = priors.recipe_from_dict(json.loads(raw))
assert restored == recipe
output = priors.generate_task(restored, seed=28, task_id=1)
expected = 2.999377735703738
assert float(torch.max(torch.abs(output.targets - expected))) < 2e-12
trace = priors.explain_generation(output, [0, 15])
(args.output / "repaired-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
assert trace["target_max_error"] < 1e-12
print("original-query-saved-generation-trace PASS", output.targets.tolist(), flush=True)
