# Quadratic functions with controlled interactions

`QuadraticRecipe` generates `quadratic-v1` tasks through the shared `generate_task`
entrypoint, saved-group runner and episode adapter. The example configuration is
`configs/generation/quadratic-cpu.toml`. Regression, binary and multiclass targets
use the same noise, separation and class-offset controls as the other families.

For output c, the sampled symmetric matrix is
`Q[j,k,c] = sum_r weights[r,c] * directions[r,j] * directions[r,k]`.
Directions are independent Gaussian draws restricted to a uniformly selected active
subset, then normalized to unit length. The `rank` setting (1–256) counts these
factors; it does not guarantee the matrix's numerical rank, especially after
changing off-diagonal entries. Each output's factor weights are independent normal
draws divided by `sqrt(rank)`. The linear weights are independent Gaussian draws,
zero outside the active subset and normalized to unit Euclidean length.

The raw function is:

```text
sum_j Q[j,j,c] * x[j]^2
+ interaction_strength * sum_{j != k} Q[j,k,c] * x[j] * x[k]
+ linear_strength * sum_j linear_weights[j,c] * x[j]
```

`interaction_strength=0` removes interactions, `1` uses the original quadratic,
and negative values reverse off-diagonal signs. Paired recipes preserve factors,
diagonal entries, linear weights and inputs. Each recipe separately estimates its
center and scale from an independent input sample. The intervention therefore
preserves the raw diagonal and linear terms, not their normalized magnitudes.
All sampled factors and normalization values are recorded in the mechanism card.
No covariance matrix per output is materialized during generation.

Inputs use the same Gaussian, uniform, Student-t and two-component Gaussian mixture
settings as the polynomial family. An unbounded Student-t requires `df > 4`, since
a quadratic needs a fourth input moment for finite signal variance. Explicitly
bounded input distributions are also supported. Overflow or degenerate calibration
fails visibly without retrying or changing the task.

## Review history

The first spec review rejected implementation `b5d731a` because it treated a
positive calibration scale below `1e-12` as degenerate. A pure quadratic with input
scale `1e-7` has the same normalized signal as scale 1, yet failed that check. The
public scale-invariance regression test reproduced the failure. The fix accepts
positive finite calibration scales and continues rejecting zero/nonfinite values.
The original rejection remains part of this record.

The standards review found no hard violations and noted one nonblocking concern:
the version decoder and generator dispatch both list families. They remain ordinary
functions for now, consistent with the specification's simple-abstraction requirement.

The follow-up spec review accepted the correction at `7b888aa` and independently
passed all seven quadratic tests. The full check at that source passed 274 tests
with one CUDA skip. Training/evaluation smoke commands exit 2 because their CLI
commands are absent.

## Reproduction

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_quadratic_generation.py tests/generation/test_family_runs.py
python3 scripts/run_check.py python scripts/quadratic_walkthrough.py --out /tmp/new-quadratic-proof
```

The walkthrough expands Q in an independent scalar calculation for one output,
then saves and resumes each task in guarded child processes. Cases cover four input
distributions, the 16-row/one-feature/rank-one lower boundary, and a combined
20,000-row/256-feature/256-factor/100-class upper boundary. These are individual
smoke and resource measurements, not stable throughput or a full Cartesian-product
coverage claim. Child peak RSS includes the interpreter and persistence; outer
elapsed time includes process startup. Tests reconstruct every output in small
regression/binary/three-class tasks, verify the interaction pair, noise/nuisance
pairs, moment rejection, replay and persistent resume.

GPU and hosted Colab acceptance for this family remain unverified. Explorer charts,
compositions, remaining families and bulk eligibility are separate pending #27 work.

The six `cpu-*.json` records identify source `7b888aa1da26d2644070d9b9006efdcdf152dc1c`.
Each saved one task without failure, then resumed with zero new generation. The
largest case used 498,315,264 bytes of child peak RSS and
1.545223 seconds for generation and persistence. Raw local
groups remain in `/tmp/r0-27-quadratic-proof-fixed/`; rerun the command above to
recreate them. These timings do not imply 100-million-task capacity.
