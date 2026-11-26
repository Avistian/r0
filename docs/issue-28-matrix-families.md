# Issue 28: recorded matrix families and kernel point laws

Implemented and independently reviewed at `7eea0bd`, following importance
checkpoint `9871fda`. This extends the accepted scope; it does not close issue 28
or replace its ledger. The evidence below is scoped to these matrix additions.

## Mathematical records

All shapes below are input-channel rows by output-channel columns, matching the
causal projection. Frozen records retain realizations; a single realization is
not a normality or independence certificate. The sampling profile defines its law.

- `GaussianWeights`: finite rectangular standard-Gaussian realizations, returned
  unchanged when authored as a raw matrix.
- `ImportanceWeights`: existing per-row positive vectors multiply Gaussian entries;
  each row divides by epsilon plus its Euclidean norm. Existing uniform and
  polynomial/log-Gaussian vector semantics remain unchanged.
- `ImportanceFactorizedWeights`: `U diag(v) V`, with a retained positive-vector
  descriptor for `v`, rather than only its realized coefficients. This delegates
  product evaluation to the existing bounded factor arithmetic. Vector values
  round to float64 before that product, including documented underflow to zero.
  The old `FactorizedWeights` record and its exploration profile remain unchanged.
- `LaplaceKernelWeights`: `sign[i,j] * exp(-distance_scale * ||left[i]-right[j]||)`.
  Points must share a dimension; signs are -1, 0 or 1, and scale is nonnegative.
  Zero scale gives the retained signs. Zero signs give exact zero. Overflow in
  an unscaled coordinate difference is handled by scaling exact differences
  before the Euclidean norm. A true large scaled distance can legitimately give
  a zero float64 kernel value. This is a cross-kernel projection, not a claimed
  positive-semidefinite Gram matrix after entry signs are applied.
- `NormalizedWeights`: add the recorded `jitter_scale * jitter`, then divide each
  row by `epsilon + norm(row)`. Jitter entries are finite recorded realizations.
  Exact affine arithmetic and scaled normalization preserve cancellation and
  finite normalized directions even when naive intermediate products overflow.
  A zero row returns zero, including epsilon zero.
- `TransformedWeights`: a normalized simple child, an existing `CausalActivation`
  applied to **one flattened row containing the entire matrix**, then recorded
  jitter. It does not standardize over dataset rows or use task labels. Channel-
  coupling activations consequently couple matrix entries, not individual matrix
  rows. The child cannot contain another transformed matrix. A top-level
  `NormalizedWeights` can wrap this result; normalization cannot directly nest
  another normalization. Decoding bounds the resulting stage depth before recursion.

Each stage materializes a finite float64 matrix independently. A nonrepresentable
raw factor/point/activation stage rejects explicitly; a later wrapper does not
retroactively rescue an invalid child stage. Parameter counts include each retained
scalar draw, explicit permutation/assignment, covariance entry, stage epsilon/scale
and activation parameter. Duplicate retained geometry is counted. Version strings,
family names and arithmetic work limits are not model scalars. Recipe geometry and
parameter admission occurs before materialization and rejects stored dense weights
that disagree with the construction.

## Kernel point constructions

`KernelPointCloud` stores centers `c_k`, Gaussian factor realizations `A_k`, positive
column-weight descriptors `v_k`, component-weight descriptor, assignments `a_r`,
innovation rows `z_r`, and log-scale parameters `mu`, `beta`, `s_k`. It computes

`point_r = c[a_r] + exp(mu + beta*s[a_r]) * A[a_r] * diag(v[a_r]) * z_r`.

For `gaussian-mixture`, innovations are Gaussian and all component parameters are
retained. Assignment probabilities normalize the retained component weights;
a zero-mass component cannot be assigned. For `random-covariance`, there is one
zero-centered component, no log scaling, and Gaussian or uniform `[-1,1)` innovations.
Gaussian factors need not be orthogonal, and their products do not certify a
numerical rank. Kernel records can retain both realized left/right points and
this construction; materialization rejects disagreement.

Weighted inner products use exact binary64 products/sums before applying the
exponential scale. The scale initially uses 80-digit Decimal arithmetic; near
center cancellation invokes higher precision. Representability bounds short-circuit
exponents beyond +/-10000: no finite binary64 triple product can compensate those
bounds. Final nonfinite points reject rather than being clipped or retried.
Positive-vector values themselves retain their previously documented float64
rounding semantics.

Each point-cloud exact-work budget counts `points * dimension**2`; each jitter
stage budget counts matrix cells; an extreme kernel-distance fallback admits the
full `rows * columns * point_dimension` bound. Defaults are 1,000,000 terms.
Positive-vector and factor-product controls retain their own limits. These are
arithmetic/geometry admission controls, not measured or OS-enforced RSS limits.

## Sampling profiles and source distinctions

Primary references: [TabICLv2 Appendix E.10-E.12](https://arxiv.org/html/2602.11139v1#A5.SS10),
pinned [`_matrix.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py),
[`_points.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py),
and [`_weights.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py).
The paper's kernel points use Gaussian innovations with random covariance. The
pinned matrix code instead uses Gaussian-mixture points; its covariance helper
also permits uniform innovations. These remain explicit distinct options here.
The matrix source uses five families, a four-family nonrecursive child selector,
child/top jitter 1e-6, transformed jitter 1e-3, and epsilon-plus-norm normalization.
Its factor construction uses `min(rows,columns)` inner dimensions, not an SVD.

`causal-matrices-v1` uses all three existing graph families and widths 1/2/4. Each
nonroot matrix independently selects Gaussian, importance, importance-factorized,
kernel or transformed with equal probability. A transformed child independently
selects one of the first four with equal probability. Every selected raw matrix
gets its own 1e-6 Gaussian jitter and row normalization with epsilon 1e-6. Transformed
raw matrices add 1e-3 Gaussian jitter after activation. Roots retain their existing
activated Gaussian form; nonroots retain default Gaussian innovations/constant noise.

`causal-gaussian-matrices-v1` selects only Gaussian raw matrices with the same
jitter and normalization. Both recipes enforce actual normalized constructions;
the Gaussian-only provenance additionally requires a Gaussian source.

The following are explicit local exploration choices, not complete source priors:

- Positive-vector family chooses uniform/polynomial with probability one half per
  matrix plan. Importance rows share that family but draw separate vectors.
  Factorized inner dimension is exactly `min(rows,columns)` with independent
  standard-Gaussian factors and the selected positive vector, without the old
  factorized profile's extra scaling.
- Kernel point law chooses mixture/covariance equally, dimension 3. Covariance
  innovations choose Gaussian/uniform equally. Mixture component count is
  `floor(exp(U*log(16)))`, U in [0,1), hence 1..15. Assignment and factor-column
  vector families are selected independently; factors share their family while
  drawing separate vectors. Centers/factors/scale parameters/innovations are
  standard-Gaussian draws where the record calls for Gaussian variables. The
  component scale is `exp(mu + beta*s_k)` with shared `mu,beta` per point cloud.
  Kernel distance scale is locally log-uniform on [0.1,10]. Entry signs are the
  signs of independent standard-Gaussian draws; exact zero is representable.
- Polynomial vector parameters use the previously retained local law, including
  decay endpoint masses. Matrix activation choices use the recipe's retained
  activation probabilities/support and independently sample their parameters.
  Node-level homogeneous parameters do not assert matrix-stage homogeneity.
- Node activation preferences, topology and task-target semantics remain as before.
  No matrix sampling law uses task observations or query labels. Plans account for
  all retained arrays before their allocation. No failure is retried or filtered.

This implements constructors and local sampling paths relevant to I2.matrix.01-.18,
including the paper/code point distinction. Source GlobalSampler correlated
numerical/categorical preferences and complete source activation weighting remain
required work; no exact source joint-prior reproduction is claimed. Wider graph,
network/tree/GP, typed-state, intervention, workbench and environment requirements
remain open. CPU backend examples do not prove browser/Colab/GPU or scientific
acceptance.

## Committed behavior checks

- Kernel geometry/generation/trace/replay: RED `c0f16b7`, GREEN `5147650`.
- Flattened rank activation and jitter: RED `9bbaa27`, GREEN `6acbfd3`.
- Retained point construction and disagreement rejection: RED `095191e`, GREEN `11ea5ea`.
- Positive-vector factor coefficients: RED `a42355e`, GREEN `7b7bb7a`.
- Both sampled profiles through three saved backend workflows: RED `a608f79`,
  GREEN `de89f5b`; the full check and independent legacy comparison below also
  cover relocation of the unchanged polynomial-vector sampling helper.


## Verification at the frozen revision

`make check` at `7eea0bd` passes **623 tests, 1 skipped**, with all static gates
passing. The [complete log](research/0028-matrices-check-7eea0bd.txt) retains the
exact output. Publication adds only documentation and verification programs;
package and test bytes remain identical to that checked revision. Reproduce with
`UV_CACHE_DIR=/tmp/r0-28-uv-cache UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check`.
The skip is the existing CUDA agreement check on this CPU-only environment.
T7/T12 train/eval commands are absent from this checkout; smoke/eval-smoke remain
NOT_AVAILABLE. These checks are not hosted CI or scientific acceptance.

The [independent spec report](research/0028-matrices-spec-review.md) found no scoped
implementation defect. Fraction/Decimal/analytic calculations pass **224 numerical
checks**, covering worked geometry, zero/cancellation/extreme inputs, normalization,
point assignments, raw-stage rejection, parameter accounting and arithmetic limits.
The separate [source review](research/0028-matrices-source-review.md) confirms the
paper/code distinctions and the explicitly adapted sampling law.

The independent sample contains **72 recipes**: **71 generated and replayed**, and
one retained generation failure (mixed profile, forward graph, multiclass, seed 1,
task 10; nonfinite square activation). All five matrix families, four transformed
child families and three point-law/base combinations occur. The
[outcomes](research/0028-matrices-outcomes.json) preserve its denominator and error;
the [artifact manifest](research/0028-matrices-artifacts.json) retains the complete
failed recipe through a header and ordered compact node arrays. Six actual saved
backend workflows cover both profiles and all graph families, with zero maximum
trace target error. This is backend evidence, not browser or Colab interaction.

Independent counting of numeric record leaves agrees with admitted parameters.
Public resource/RNG instrumentation and source inspection verify admission before
new matrix-array allocation. **378 legacy outcomes** across all six earlier profiles
match exactly: 376 successes and two preserved failures. The independent runner
uses one OpenMP/MKL thread for both archived revisions; it records actual imports,
archive hashes, commands and script hashes. This is exact equality in that declared
environment, not a cross-platform determinism claim.

The first oracle attempt incorrectly expected a factor exact-work rejection when
its positive coefficient had rounded to zero, making fallback unnecessary. That
[harness error](research/0028-matrices-harness-error.json) remains separate from the
successful corrected run. No candidate source changed to satisfy that correction.
[Standards review](research/0028-matrices-standards.md) found no hard violations;
duplicated component-probability normalization is a nonblocking maintainability
heuristic. Standards and numerical/spec findings are separate axes.

The [receipt](research/0028-matrices-receipt.json) and
[final package check](research/0028-matrices-final-package.json) bind the published
programs to their independently executed bytes. The ordered artifact manifest
preserves all 224 numeric records, admission events, legacy outcomes and the
independent failed recipe; concatenation reconstructs each original value exactly.
Raw per-case recipes, output and saved archives remain in
`/tmp/matrix-spec-7eea0bd/run-2/`; the earlier harness run remains in `run-1/`.

The portable [runner](../scripts/verify_causal_matrices.py),
[numerical cases](../scripts/causal_matrix_numeric.py),
[sampling/workflow cases](../scripts/causal_matrix_cases.py), and
[legacy recorder](../scripts/causal_matrix_legacy.py) require no retained `/tmp`
inputs. From a checkout containing both pinned revisions and a compatible existing
environment, use a new output directory:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_matrices.py \
  /tmp/r0-28-matrices-independent-reproduction --repo .
```

Defaults pin candidate `7eea0bd` and base `9871fda`. The runner archives both, makes
the archives read-only, sets their import paths, and retains all check outcomes.
Only override `--candidate`/`--base` deliberately when assessing other revisions.

## Unfiltered author censuses

Each census attempts 64 recipes per graph family at seed 28, cycling regression,
binary and multiclass controls. Recipes are captured in memory before generation;
JSONL is written after each family completes, not durably before every attempt.
Both retain all failures with no retry, replacement or selection filter.

- **Mixed profile: 192 valid / 192 attempted.** Its 1,523 top-level projections
  contain 1,840 normalization stages, including 317 transformed children. Across
  all stages there are 367 Gaussian, 388 importance, 377 importance-factorized and
  391 kernel records. Kernel records split into 193 mixture and 198 covariance
  point clouds, with component counts 1..15 observed. The existing 26-choice matrix
  activation support appears in full. These are record counts, not numerical ranks
  or proof of complete source activation weighting. Valid flags overlap: 73
  disconnected, 20 inactive ReLU, 23 saturation and three zero target dispersion.
  [Summary and raw hashes](research/0028-matrices-mixed-census.json).
- **Gaussian-only: 190 valid / 192 attempted.** All 1,591 retained projections are
  normalized Gaussian constructions. Forward task 57 (regression) fails in signed
  power; task 58 (binary) fails in integer power. Both full failed recipes are
  retained and reconstructible from the [summary and ordered parts](research/0028-matrices-gaussian-census.json).

Raw JSONL and saved groups remain at `/tmp/r0-28-matrix-census-7eea0bd/` and
`/tmp/r0-28-gaussian-matrix-census-7eea0bd/`. Timing and peak memory are NOT_MEASURED.
Reproduce each in a new output directory using:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/causal_topology_census.py \
  /tmp/r0-28-matrices-census-reproduction --profile causal-matrices-v1
```

For the other census, change both the destination and profile to
`causal-gaussian-matrices-v1`. Whole-issue, GPU, supported-size, browser/Colab,
source joint-prior and scientific acceptance remain incomplete.
