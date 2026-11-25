# Additive polynomials and independent input distributions

`PolynomialRecipe` generates sparse additive polynomials of degrees 1–16. Degree 1
also supplies linear tasks on the newly supported input distributions. The older
`linear-v1` Gaussian recipe and its exact variance normalization remain unchanged.
These new recipes normalize using an independent population sample; their unit
variance is an estimate, not an exact guarantee or bulk-training qualification.

Each active feature receives a power sampled uniformly from 1 through the configured
degree. One active feature is assigned that degree, so the requested order is present.
Output weights are independent Gaussian draws, masked to the active subset and scaled
to unit Euclidean norm. For output c, the raw function is `sum_j weights[j,c] * x_j**powers[j]`.
Subtract the stored center and divide by the stored scale. The card retains powers,
weights, active roles, normalization values and the complete input distribution.

## Inputs and moments

Columns are independent. Before `location + scale * value`, their distributions are:

- Gaussian: standard normal.
- Uniform: [-sqrt(3), sqrt(3)].
- Student-t: the declared positive degrees of freedom, using the usual t scale.
- Mixture: equal-weight Gaussians with unit component variance and means at
  `-mixture_gap` and `+mixture_gap`.

`scale` is a multiplier, not a claim that every distribution has that standard
deviation. An optional symmetric `bound >= 0.1` conditions the base distribution
on `abs(value) <= bound` before location/scale. Rejection sampling implements this
condition; no values are clipped. Each column allows at most 128 batches, each with
`max(rows, 256)` candidates. Exhaustion is an explicit failure, not a replacement task.

For unbounded Student-t inputs, finite polynomial variance requires `df > 2*degree`.
The t density's tail is proportional to `abs(x)**(-df-1)`, so its moment of order m
exists only when `df > m`. The recipe checks this condition before sampling. A
finite empirical variance does not override it. Bounded inputs have all moments,
but overflow or degenerate calibration still fails explicitly. Very high degrees
and heavy tails can make estimates unstable even when moments exist; these settings
remain exploration cases pending coverage and resource qualification.

## Reproduction and evidence

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_polynomial_generation.py
python3 scripts/run_check.py python scripts/polynomial_walkthrough.py --out /tmp/new-polynomial-proof
```

Choose a new output directory. `cpu.json` retains the source revision, requests,
independent scalar calculations and guarded-run reports. The six cases cover all
four input distributions, a bounded degree-16 task and 20,000 rows × 256 features.
Each saves one task; raw local groups from the recorded run remain under
`/tmp/r0-27-polynomial-proof/`. They can be regenerated with the retained command.
Timing includes generation, independent calibration, serialization and persistence;
outer elapsed time additionally includes process startup. Child lifetime peak RSS
is recorded. One task per case is smoke/resource evidence, not a stable throughput
estimate or a full Cartesian-product validation.

Public tests initially failed on the missing interface. They independently reconstruct
rows, reject the moment boundary, distinguish truncation from clipping, replay cards,
preserve noise/nuisance pairs and retain absent-class tasks. The runner and admission
use the common decoder; diagnostic recipes preserve the selected inputs and degree.
The shared output helper retains existing Fourier behavior. GPU/Colab verification,
charts, quadratic interactions, other families and bulk eligibility remain pending.
