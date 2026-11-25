# Valid polynomial input scales

The quadratic review exposed the same absolute calibration cutoff in polynomial
sampling. Before this fix, the public scale-invariance test passed for degree 1
but rejected degrees 4 and 16 at input scale `1e-7`, even though their standard
deviations were positive, finite and representable. Multiplying a one-feature
monomial input by a positive number multiplies both raw signal and calibration
scale by that number raised to the degree; normalization cancels it.

The fix rejects zero or nonfinite calibration scales, without the arbitrary
`1e-12` floor. Existing successful tasks retain their sampled parameters and
outputs. Actual floating-point underflow to zero still fails visibly.

## Independent row reconstruction

Run from the repository using the existing environment (no network or dependencies):

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py python - <<'PY'
import json
import math
from r0.modules.priors import InputDistribution, LinearRecipe, PolynomialRecipe, generate_task

for degree in (1, 4, 16):
    recipe = PolynomialRecipe(
        task=LinearRecipe(features=1, signal_features=1, active_features=1),
        inputs=InputDistribution(scale=1e-7), degree=degree,
    )
    task = generate_task(recipe, seed=27, task_id=1)
    card = json.loads(task.mechanism_json)
    x = float(task.inputs[0, 0])
    weight = card['weights'][0][0]
    raw = weight * x**degree
    expected = (raw - card['center'][0]) / card['scale'][0]
    actual = float(task.scores[0, 0])
    assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
    print(json.dumps(dict(degree=degree, x=x, weight=weight,
                         center=card['center'][0], scale=card['scale'][0],
                         scalar_expected=expected, scalar_actual=actual)))
PY
```

Observed output:

```jsonl
{"degree": 1, "x": -3.0341352667143855e-08, "weight": 1.0, "center": 7.931526048683993e-10, "scale": 9.899638307814785e-08, "scalar_expected": -0.31450144241567535, "scalar_actual": -0.31450144241567535}
{"degree": 4, "x": -3.0341352667143855e-08, "weight": 1.0, "center": 2.9024238984206485e-28, "scale": 8.96672325190029e-28, "scalar_expected": -0.32274319349516495, "scalar_actual": -0.32274319349516495}
{"degree": 16, "x": -3.0341352667143855e-08, "weight": 1.0, "center": 4.6320756522697453e-107, "scale": 8.012537515393877e-106, "scalar_expected": -0.05781034589067995, "scalar_actual": -0.05781034589067995}
```

The calculation uses the retained scalar input, coefficient, center and scale.
Public tests also compare all normalized scores against the unit-input-scale task.
These are CPU correctness checks, not GPU or bulk-throughput evidence.
