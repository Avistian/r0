# Additive and projected smooth functions

`SmoothRecipe` samples sinusoidal or Gaussian radial-basis terms. Each term acts on
one feature (`projection="additive"`) or a weighted feature combination
(`projection="ridge"`). A ridge function here means a scalar function of a linear
projection. This gives four named variants through the same generator and runner.

For term r and output c, the raw function is:

```text
z[r] = (sum_j directions[r,j] * x[j] - centers[r]) / widths[r]
basis[r] = sin(z[r])                       # sinusoidal
basis[r] = exp(-z[r]^2 / 2)                # radial
raw[c] = sum_r weights[r,c] * basis[r]
score[c] = (raw[c] - center[c]) / scale[c]
```

The radial basis is one-dimensional, applied to a feature or projection. The
sinusoidal term has angular frequency `1/width` and phase `-center/width` in that
coordinate. Parameters and raw formula are retained for later response-curve and
intermediate-value inspection.

## Parameter distributions and constraints

The active subset is a uniformly permuted selection of signal features. Additive
terms cycle through that subset, with one unit coefficient per direction; there
must be at least one term per active feature. Ridge directions use independent
standard normal coordinates, zero inactive coordinates, and unit Euclidean length.
Each ridge term therefore uses the selected features together.

`terms` ranges from 1 through 4,096. These are scalar basis terms inside one sampled
function; the later composition limit counts combined generator functions instead.
Widths are log-uniform between `length_scale / width_spread` and
`length_scale * width_spread`, with positive finite `length_scale` and finite
`width_spread >= 1`. A spread of 1 fixes every width. A term's center is
`inputs.location * sum(direction) + inputs.scale * Normal(0,1)`. This uses the
input distribution's affine location/scale parameters, not a claim about its mean
or variance. Per-output weights are independent standard normal draws divided by
`sqrt(terms)`. Their randomness is independent of the input, noise and split streams.

Both bases are bounded, so a sampled finite weighted sum has finite signal variance
even for Student-t inputs without a second moment. All four input distributions
are supported. Calibration uses a separate reproducible sample and CPU arithmetic,
then the actual task's raw scores are computed on the selected CPU/CUDA device.
Normalization is estimated, not an exact unit-variance guarantee. Nonfinite sampled
parameters or transformed arguments, and degenerate/nonfinite calibration, fail
visibly without clipping, substitutions or retries. Extreme configurations are
exploration cases, not automatically eligible for bulk training.

The evaluator handles 256 rows at a time, so the temporary basis matrix does not
grow to `task.rows × terms`. Full requested task arrays and compact function
parameters remain in memory. Noise/nuisance pairs preserve the function, shared
input columns and split. Cards retain directions, centers, widths, weights, active
roles and calibration values; saved recipes replay through the common decoder.

## Reproduction

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_smooth_generation.py tests/generation/test_family_runs.py
python3 scripts/run_check.py python scripts/smooth_walkthrough.py --out /tmp/new-smooth-proof
```

The walkthrough saves tasks in guarded children and reconstructs one scalar output
from each persisted input/truth archive and parameter card. It records the checked
row index and archive receipt, then verifies zero-generation resume. Eight cases
cover four variants, four input distributions, all three task types, the smallest
shape, a medium ridge function and both large additive/ridge examples with 20,000
rows, 256 active features and 4,096 terms. The large radial ridge case also has
100 classes. They are individual CPU smoke/resource cases, not stable throughput
estimates or full Cartesian-product validation. Requests declare 120 seconds and
256 MiB output limits each; timings include generation/persistence and separately
record startup-inclusive outer time. Child peak RSS includes its interpreter.

The initial public test failed on the missing interface. The full check passed
343 tests with one CUDA skip; the retained script then passed format, lint, typing
and repository guards. Training/evaluation smoke commands exit 2 because the
train/eval CLI commands are absent, so no loss or evaluation table exists. GPU,
hosted Colab, explorer charts, remaining families and final bulk coverage remain
pending.

## Retained measurements and reviews

All eight `cpu-*.json` records identify source `ae212c791036906299cd0867970147c5d4450906`.
Each saved one task without failure and resumed with zero generation. The largest
radial ridge case used 616,906,752 bytes of child peak RSS and 5.809371 seconds
for generation/persistence. The large additive sinusoidal case used 440,193,024
bytes and 3.315096 seconds. The records contain archive SHA-256 and byte sizes,
binding the scalar calculations to the saved data. Raw groups remain at
`/tmp/r0-27-smooth-proof/`; the retained command recreates them.

Both independent implementation reviews found no blocking finding. The spec
reviewer passed 39 focused public tests; the standards reviewer noted repeated
family-dispatch lists as an optional maintenance concern. Neither code review
claimed GPU or wider issue acceptance.
