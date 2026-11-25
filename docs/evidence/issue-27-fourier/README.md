# Periodic and random Fourier functions

`generate_fourier(FourierRecipe(...), seed=..., task_id=..., device=...)` returns
the existing `TrainingTask` with separate predictor input and hidden truth.
`task=LinearRecipe(...)` currently supplies shared sizes and target controls;
it does not select the nonlinear mechanism. Runner configuration integration and
explorer rendering follow separately; this slice does not complete either family.

For RFF, each frequency vector has independent Gaussian entries with standard
deviation `1 / length_scale`, restricted to sampled active features. For periodic
mixtures, each component uses one active feature, cycling through the active set.
Base angular frequencies are uniform on [0.25, 4] divided by `length_scale`;
successive components on a feature use integer harmonics, preserving its period.
Phases are uniform on [-pi, pi], and output coefficients are Gaussian with standard
deviation `1 / sqrt(terms)`. All draws have explicitly keyed streams. Supported
term counts are 1–4096 for RFF and 1–64 for periodic mixtures. Periodic mixtures
require at least one component per declared active feature.

For a row x and output c, the raw function is
`sum_j coefficients[j,c] * cos(dot(x, frequencies[j]) + phases[j])`.
Subtract the stored `center[c]` and divide by `scale[c]`. These are estimated from
an independent Gaussian sample, default 4096 rows, not from context or query rows.
The estimate is not an exact unit-population-variance guarantee. Finite trigonometric
functions have finite moments under these inputs. Degenerate/nonfinite calibration
fails without retry. No exact-GP distribution claim is made for finite RFF functions.

The card retains all frequencies, phases, coefficients, active roles and normalization
values. The same card supports actual analytic response curves in the planned explorer.
Noise, class probabilities/labels and splits use the existing linear target adapter,
now shared as an ordinary private function. Existing linear regression checks remain
unchanged. Nuisance-column and noise interventions preserve the hidden function.

Reproduce local evidence using the existing verification environment:

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_fourier_generation.py tests/priors/test_linear_generation.py
python3 scripts/run_check.py python scripts/fourier_walkthrough.py
```

The public-interface tests failed on missing exports before implementation. They now
reconstruct a target with scalar `math.cos`/`math.fsum`, replay tasks, compare pairs,
verify calibration is independent of row count and retain weak classification tasks
with 2, 3 and 100 classes. The walkthrough checks a row independently and measures
three new task identities after a separately reported warmup. Timings include sampling,
calibration and task construction, but exclude episode conversion, archive saving and
process startup. Memory is **NOT_MEASURED**. Torch computation uses bounded 256-row
chunks; peak memory still requires measurement before bulk eligibility claims.

The retained CPU cases cover a periodic mixture, 256-term RFF, 20,000 rows with 256
features, and a small 4096-term task. This is not validation of every parameter
combination. GPU verification of these new functions remains **NOT_RUN**; earlier
hosted linear GPU receipts do not establish it. Additional feature distributions,
compositions, exact GPs and other approved families remain separate work.
