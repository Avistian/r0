# Direct Gaussian-process reference tasks

`GaussianProcessRecipe` draws values jointly at independent calibration inputs and
task inputs using a dense covariance matrix. It supports RBF and Matérn 1/2, 3/2
and 5/2 kernels, weighted sums, products, feature-specific length scales and kernels
on selected subsets of active features. This is an expensive reference sampler:
16–1,024 task rows, 32–512 calibration rows, 1–256 features and up to 100 classes.
The joint support has at most 1,536 points. It does not substitute Fourier features.

## Covariance and parameters

For distance r in length-scaled feature coordinates, the four kernels are
`exp(-r²/2)`, `exp(-r)`, `(1+sqrt(3)r)exp(-sqrt(3)r)` and
`(1+sqrt(5)r+5r²/3)exp(-sqrt(5)r)`. These formulas and the sum/product constructions
are implemented independently from [GPML chapter 4, equations 4.9 and 4.17 and
section 4.2.4](https://gaussianprocess.org/gpml/chapters/RW4.pdf). A product of
covariance kernels defines a GP covariance; multiplying independent sampled GP
functions is a different, generally non-Gaussian construction.

Each `StationaryKernel` specifies its kind, length scale, optional feature-scale
multipliers, optional axes and positive weight. Axes index positions in the sampled
active-feature list, not the original input columns. An empty axes tuple selects all
active features. Components together must cover every declared active feature.
An empty feature-scale tuple means unit multipliers; otherwise its length must match
the component axes. This permits additive subset GPs and anisotropic covariance.
Kernel hyperparameters are fixed by the recipe; the active subset and function draw
are sampled. Explicit exploration mixtures remain separate from production weights.

There may be 1–64 kernel components. A sum normalizes positive weights to sum to one,
using rescaling to avoid overflow; weight underflow fails rather than silently
removing a component. A product uses unit-weight components and retains unit diagonal.
No jitter or undeclared white-noise component is added.

## Joint draw, precision and normalization

The sampler coalesces duplicate active-coordinate inputs before drawing, so repeated
inputs share latent scores. It computes the symmetric covariance square root
`F = V sqrt(max(lambda, 0)) V.T` on CPU. This removes arbitrary eigenvector signs
and rotations. Independent standard-normal vectors z then give raw values `F z`.
Normal vectors are sampled separately per output and retained. Task multiplication
runs on the selected CPU/CUDA device; covariance construction, factorization and
calibration remain CPU work. GPU acceleration of the factorization is not claimed.

The declared roundoff tolerance is
`64 * float64_epsilon * support_count * max(1, largest_eigenvalue)`. Eigenvalues below
the negative tolerance fail. Smaller negative eigenvalues are set to zero, and the
maximum entrywise error of `F F.T` versus the requested covariance must fit the
same tolerance. Positive eigenvalues are retained. The card records the smallest
eigenvalue, correction size, tolerance and measured covariance error. Thus “direct”
means the full finite covariance is used, subject to recorded floating-point error;
it is not a claim of real-arithmetic exactness.

Calibration variation must exceed covariance roundoff resolution. Its expected
empirical variance under the GP is the mean calibration diagonal minus the mean
calibration covariance. Near-constant covariance that float64 cannot resolve fails
explicitly. No resampling, kernel replacement or size reduction follows a failure.

The same joint draw supplies independent calibration inputs and task inputs. The
sampled function's calibration mean and standard deviation normalize its task
scores. Calibration is independent of the observation inputs and target noise,
but its function values are intentionally correlated through the same GP. The raw
draw is Gaussian; dividing by a sampled calibration scale changes its marginal
prior distribution. Unit signal variance is an estimate, not an exact guarantee.
All four input distributions are accepted, including heavy tails: these bounded
covariance kernels have finite diagonal variance without input-moment assumptions.

Cards retain active features, all kernel settings, unique support, calibration/task
support mappings, Gaussian vectors and normalization statistics. They retain no
dense covariance/factor trace: those can be reconstructed for selected inspections.
The record describes original sampled values. Interpolation and additionally
sampled values will need explicit labels in the explorer.

## Reproduction and scope

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_exact_gp.py tests/generation/test_family_runs.py
python3 scripts/run_check.py python scripts/exact_gp_walkthrough.py --out /tmp/new-gp-proof
```

The initial public test failed on the missing interface. Additional RED checks
established the symmetric-root draw contract and visible kernel-weight underflow.
Public checks cover known identity covariance, independently calculated subset
sum/product covariance across all target types, replay, duplicate inputs, paired
noise/nuisance controls, reference limits and unresolved covariance variation.

The walkthrough uses an independent squared-distance identity and elementary kernel
formulas to reconstruct covariance and one saved scalar target (`1e-8` tolerance).
Eight success cases cover all four kernels/input distributions, three task types,
a subset sum, the minimum shape, 64 components and a maximum-shape product with
1,024 rows, 512 calibration points, 256 active features and 100 classes. It also
retains an intentional near-constant-covariance failure. Successful cases verify
saved-group checksums and zero-generation restart. Child RSS and generation/
persistence time are separate from outer startup-inclusive time. These are
single-task CPU checks, not stable throughput or bulk-training qualification.

Full checks at the corrected source passed 361 tests with one CUDA skip; format,
lint, typing, boundaries and repository guards passed. Exact measurements and
review outcomes follow below.
GPU/hosted Colab acceptance, explorer views, remaining kernels/compositions and
final #27 coverage remain pending. Training/evaluation CLI commands are still
absent, so no training-loss summary or evaluation table is available.

## Review history

The first spec review rejected source `f03a1c7`. Subtracting opposite-sign finite
coordinates before dividing by a large length scale could overflow, producing zero
covariance where the correct normalized distance was finite. Subsequent matrix
checks validated that incorrect covariance and did not detect the error. Four new
public tests reproduced the defect across all kernels with uniform input scale
`8e307` and length scale `1e308`. The fix rescales before subtraction only when
the raw subtraction overflows, preserving the original arithmetic otherwise.
The independent oracle uses scaled coordinates and recomputes both covariance and
calibration; its output tolerance is `1e-10`. The original rejection is retained.

The correction review accepted `2374d8d` and independently passed 17 GP tests.
The original four-dimensional adversarial case's error fell from `0.299447999`
to `6.88e-15`. Standards review found no hard violation and noted repeated family
dispatch lists as a nonblocking maintenance concern.

All retained receipts identify source `2374d8de7dc9b11abde3690835e73fa806fde54c`.
Eight cases each saved one task without failure and resumed without generation.
The largest product case used 439,820,288 bytes of child peak RSS
and 9.637754 seconds for generation/persistence. Archive
SHA-256/size receipts bind each scalar calculation to its saved dataset. Raw groups
remain under `/tmp/r0-27-gp-proof-fixed/`; the retained command recreates them.
The ninth run failed as intended with an explicit covariance-resolution error,
retaining task ID 0 and its request in `failure.json`. This is not acceptance of
a 100-million-task run or qualification of direct GPs for bulk generation.
