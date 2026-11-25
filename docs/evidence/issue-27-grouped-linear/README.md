# Grouped-sparse linear functions

`GroupedLinearRecipe` selects complete groups of signal features. Adjacent feature
indices form equal-sized groups: with eight signal features and `group_size=2`,
the groups are `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`. The generator samples a uniform
permutation of group indices and takes `active_features / group_size` groups.
Every feature in a selected group is active; other coefficients are zero.

This version supports group sizes 1–256. Both `signal_features` and
`active_features` must be multiples of the group size; partial groups are rejected.
Group size 1 gives ordinary feature sparsity, and selecting every group gives a
dense linear function. This is an explicit equal-sized, non-overlapping partition,
not a claim to support arbitrary overlapping or unequal groups.

Per-output coefficients are independent standard normal draws before masking,
then normalized to unit Euclidean length. Group membership is shared across
outputs; coefficients are sampled independently for each output. Inputs use the
same configurable Gaussian, uniform, Student-t and two-Gaussian-mixture sampler as
the polynomial/quadratic families. Unbounded Student-t inputs require `df > 2` for
finite signal variance; bounded versions permit smaller positive degrees of freedom.

The raw function is `sum_j weights[j,c] * x[j]`. Signal normalization subtracts
the stored center and divides by the stored scale, estimated from a separate
reproducible input sample. This is population calibration, not preprocessing fitted
to query labels. Positive finite small scales remain valid. Nonfinite values and
zero calibration scales fail visibly without replacing or retrying the task.

Cards retain all groups, selected group indices, active feature indices,
coefficients, calibration values and input distributions. A fixed seed/task ID
preserves the mechanism when observation noise or trailing nuisance columns change.
The same public generator and episode format feed the bounded runner, saved groups
and resume checks. `recipe.group_size` is features per group;
`run.group_size` is tasks per saved archive and has a different purpose.

## Reproduction

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_grouped_linear.py tests/generation/test_family_runs.py
python3 scripts/run_check.py python scripts/grouped_linear_walkthrough.py --out /tmp/new-grouped-proof
```

The walkthrough independently sums one output's scalar target, checks whole-group
activation, saves each task, then resumes with no new generation. Seven cases cover
all four input distributions, all three task types, 16 rows/one feature, a medium
2,048-row/64-feature sparse task and a 20,000-row/256-feature/100-class dense task
with one group of 256 features. These are single-task smoke/resource measurements,
not stable throughput or full Cartesian-product coverage. Each record includes
child lifetime peak RSS, generation/persistence time and outer startup-inclusive
time. The 100-million-task scale is not validated by these measurements.

The initial public test failed because `GroupedLinearRecipe` was missing. The full
check passed 301 tests with one CUDA skip; the retained script subsequently passed
format, lint, typing and repository guards. `make smoke` and `make eval-smoke` exit
2 because train/eval CLI commands are absent; no training loss or evaluation table
was produced. GPU and hosted Colab acceptance, charts and remaining #27 families
are still pending.

## Review history

The first spec review rejected source `21c71fe`: direct variance estimation
squared the raw calibration values, causing underflow at input scale `1e-200`
and overflow at `1e200`. The normalized linear functions remained representable.
Six new public cases reproduced this failure across group sizes 1, 4 and 8.
The correction divides calibration values by their per-output maximum magnitude
before estimating mean/standard deviation, then restores units. It still rejects
zero or nonfinite calibration. The historical rejection remains visible here.

The follow-up spec review accepted the correction at `366b081`, independently
passed all 23 grouped-linear tests, and checked both extreme scales against the
unit-scale normalized signal (maximum difference `6.66e-16`). Standards review
found no hard violations or actionable smells.

The seven `cpu-*.json` records identify source `366b0816f06bede83d8fe2846076695d2814e8f1`.
Each saved one task without failure, then resumed with zero new generation.
The largest case used 458,993,664 bytes of child peak RSS and
0.920387 seconds for generation and persistence. Raw groups
remain at `/tmp/r0-27-grouped-proof-fixed/`; rerun the retained command to recreate
them. This is not evidence of full training, hosted GPU or 100-million-task capacity.
