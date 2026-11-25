# Controlled pairs: generation and model evidence

Source: `d6502930a072422ec4fa6fd708b2555f466ee1b2`. The walkthrough ran from
that committed source and wrote `/tmp/r0-pairs-d650293`. `files.json` records every
raw artifact's size and SHA-256 hash. The raw summary is split losslessly into
`summary.json` metadata and three ordered 30-case files. `ols-records.csv` retains
all fields from the raw OLS record JSON. Selected cards, one regression noise
pair's arrays per family, and all OLS query-error vectors are committed here.

## Generation coverage

The 90 cases cover all ten starting families, all three task types and three
interventions per task type: regression noise or classification separation,
irrelevant columns, and sparsity. Each pair verifies its declared fixed properties;
both tasks replay directly from their card records with identical episodes and
mechanism JSON. Regression noise residuals scale by four. Zero class separation
produces the independently expected uniform probabilities.

Gaussian-linear sparsity pairs preserve expected unit variance analytically.
The other sparsity cases **explicitly opt into calibration-based estimates**.
Those cases do not prove equal true expected variance, and the default API rejects
them without that choice. Cards retain the variance policy, actual center/scale,
realized signal spread, identities, both replay records and measured target changes.
Invalid geometry and unrequested recipe changes remain explicit errors.

## Fixed-model measurements

One ordinary least-squares model with intercept was fixed in advance. It reads only
the public `Episode.predictor_view()` for fitting and prediction: context rows and
labels for the fit, query inputs for prediction. Observed query targets are used
only to calculate errors afterward. There is no tuning or model selection.

The three interventions share the same 30 root draws of Gaussian linear tasks:
256 rows, 192 context rows, 16 features, eight active features, noise ratio 0.2.
These are 30 shared roots across interventions. The NPZ retains all 64 query errors
for both sides of every pair.

| Intervention | Change | Mean query MSE before | Mean query MSE after |
| --- | --- | ---: | ---: |
| Noise | Ratio 0.2 → 0.8 | 0.043495 | 0.695923 |
| Irrelevant columns | 16 → 64 total features | 0.043495 | 0.057737 |
| Sparsity | 8 → 2 active features | 0.043495 | 0.043495 |

These outcomes concern this fixed model and cohort. Sparsity did not increase
difficulty for this model; no monotonic difficulty assertion is imposed on every
model or individual draw. This is not frontier-model or full-training evidence.

Recompute the table directly from retained error vectors:

```python
import numpy as np

with np.load(
    "docs/evidence/issue-27-controlled-pairs/ols-query-errors.npz", allow_pickle=False
) as archive:
    for intervention in archive.files:
        errors = archive[intervention]
        assert errors.shape == (30, 2, 64)
        print(intervention, np.mean(errors[:, 0] ** 2), np.mean(errors[:, 1] ** 2))
```

Reproduce generation, replay and model measurements in a fresh directory:

```bash
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/controlled_pair_walkthrough.py \
  --out /tmp/r0-pairs-reproduction
```

Choose another unused output name if the example directory already exists.

Per-case elapsed times are retained as run observations, not throughput targets or
dedicated resource benchmarks. Full checks were running concurrently. GPU and
hosted-runtime evidence are not inferred from this CPU run.

## Checks and limitations

Before implementation, tests failed on the absent public pair API and unsupported
interventions. The first walkthrough completed generation cases but failed because
its OLS consumer used nonexistent Episode field names. The corrected procedure
uses the actual public predictor view. Review also found that cards initially
described normalization without retaining center/scale; a missing-field test
failed before those values were added. The final source resolves that finding.

At the recorded source, `make check` passed **462 tests, 1 CUDA skip**, in 101.41 s.
Then `make smoke` and `make eval-smoke` each exited 2 because the CLI has no `train`
or `eval` command; no training loss summary or evaluation table exists. Both
source-review axes have no remaining blocking findings.

This slice delivers the public pair API and CPU evidence. The side-by-side explorer,
persistent pair exports, target-runtime checks and remaining #27 scope are pending.
No cloud run was launched, and this PR does not close #27.
