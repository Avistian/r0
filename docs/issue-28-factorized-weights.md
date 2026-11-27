# Issue #28 factorized structural projections

This is a partial weight-family checkpoint under accepted decisions 10, 13 and 18
of the [full specification](specs/0028-causal-network-generators.md). It does not
complete weight sampling, neural architectures or the paper-option inventory.

## Frozen construction and evaluation

`FactorizedWeights` records finite rectangular matrices `left`, `right` and a
nonnegative `diagonal` vector. The inner dimensions must agree. Its equation is

```text
W = (left * diagonal[None, :]) @ right
node_affine = input_activation(parent_channels) @ W + bias
```

`MatrixCausalNode` extends the existing stochastic activated node and stores both
the realized dense `weights` and its `weight_construction`. Ordinary causal nodes
keep their existing fields and RNG identities. A matrix node must have parents;
its recorded projection dimensions must match the dense weights and output width.
Recipe admission counts the construction scalars in addition to the dense matrix.
Only after admission does recipe validation materialize the factors and require
exact agreement with the stored dense projection. Generation uses the admitted,
frozen dense weights; it does not resample or refit the matrix from task rows.
Nonfinite construction results fail explicitly without retry or clipping.

The public decoder restores nested construction records. Original-row inspection
shows both the effective weights and their factors. Saved recipe replay retains
these fields and identities. Removing a required projection record from the
`activation-factorized-v1` sampled construction is rejected. Existing Gaussian
innovations, signal-dependent responses, activation stages and moment checks
remain available on authored matrix nodes.

A worked example is

```text
left = [[1, 2], [3, 4]]
diagonal = [2, 0.5]
right = [[5, 6], [7, 8]]
W = [[17, 20], [44, 52]]
first output, zero bias/noise = 17*x0 + 44*x1
```

The retained test also checks the complete parameter count: four root scalars,
eight dense child scalars, seven default exogenous/response scalars and ten
factor scalars, 29 total. These are admission/storage estimates, not measured
peak memory. The factors are not required to be orthogonal, and `diagonal` is not
claimed to contain actual singular values. Zero coefficients and rank-deficient
projections remain valid.

## Versioned exploration profile

`causal-factorized-v1` uses forward, layered or predecessor-redirection topology,
node widths uniformly drawn from 1, 2, 4, and the existing 26-choice activation
preference law. Roots retain the existing Gaussian innovation law. For each
nonroot with `n` incoming channels and `m` output channels:

- Choose inner dimension `k` uniformly from integers 1 through `min(n,m)`.
- Draw independent standard Gaussian `left[n,k]` and `right[k,m]`.
- Set every inner coefficient to `1/sqrt(n*k)`.
- Store the factors and their dense product on the node.

Each entry has population variance `1/n` under this law; entries are not generally
independent. Biases and base structural-noise scales retain the existing Gaussian
affine sampler's settings. All factor dimensions and their added storage are
admitted before factor arrays are allocated. The profile does not retry failures,
filter valid draws or alter previous profiles. It reaches the shared explorer
backend's generation, selected-row trace, save/reopen and replay paths. Actual
browser and Colab interactions remain pending.

## Source distinctions and remaining scope

The [pinned I2 matrix source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py)
forms Gaussian factor products with inner positive weights and explicitly avoids
an orthogonality claim. This checkpoint delivers the factor-product construction
and interpretation relevant to `I2.matrix.03` and `I2.matrix.18`. Its configurable
authored coefficients and separate r0 exploration profile are not an identical
source sampler: the source uses inner dimension `min(n,m)`, sampled positive
weights, top-level jitter and row normalization. Those sampling details remain
required, separately from this construction.

The [positive-weight source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py)
also distinguishes uniform values from shuffled polynomial decay with log-Gaussian
perturbations. Those laws, importance-weighted Gaussian matrices, signed kernel
matrices and their distinct point variants, activation-transformed matrices,
nonrecursive children, jitter/normalization stages, masks/block sparsity and
shared weight-family preferences remain required. Source inspection did not copy
foreign implementation code. The full #28 goal stays active.

## Development and verification

The authored construction has RED `04f8e1c`, GREEN `559d2af`; saved sampled workflow
has RED `a14f620`, GREEN `fa26bcb`; discarded-provenance rejection has RED `dc84af8`, GREEN `40e002c`.
The initial authored test contained an arithmetic typo in the unused second output
(50 instead of 52) and expected a list where inspection retains a tuple. Both were
corrected; the target's independently specified `17*x0 + 44*x1` stayed unchanged.
An initial commit hook also caught import ordering; no hook was bypassed.

The focused authored checks and all three new saved-workflow cases pass. Exact
frozen revisions, independent review, full checks and census results are appended
after execution; this paragraph does not claim those gates already passed.


## Numerical repair and bounded exact arithmetic

Independent review rejected `ec522ed`: `left=2**1023`, coefficient 4 and
`right=2**-1023` should produce the finite projection 4, but the intermediate
left/coefficient product overflowed. The [original report](evidence/issue-28/legacy-research/0028-factorized-original-review.json),
[exact input](evidence/issue-28/legacy-research/0028-factorized-original-failure.json) and
[revision receipt](evidence/issue-28/legacy-research/0028-factorized-original-receipt.json) retain the
failure. Ten earlier authored/validation probes passed; the reviewer stopped
at the defect, so that review did not establish sampled-law, backend-workflow
or legacy-profile acceptance.

RED `209b4e6` retains overflow, underflow and signed cancellation cases.
Repair `ef1bacc` conservatively checks the binary64 factors' exponent ranges.
Ordinary products keep their existing NumPy path. Potentially overflowing or
subnormal intermediates use exact Decimal representations of the input floats,
streaming triple products into a sufficiently wide sum, then converting each
final entry to float64 once. The precision bound allows three factors each with
up to 309 integer digits and 1074 fractional places plus the sum's carry digits.
The fallback does not retain another complete set of Decimal factors. It also
handles a nonfinite result detected on the ordinary path.

The explicit `exact_terms_limit` defaults to 1,000,000 terms and is configurable
per construction. The bound counts `n*m*k` triple-product summands; it applies
only when exact arithmetic is needed. Exceeding it is a declared resource failure.
A genuinely nonrepresentable final entry remains an explicit numerical failure.
Positive mathematical values that round to zero at float64 precision are distinct
from losing a representable result in an intermediate calculation. This mechanism
does not promise exact real arithmetic for all ordinary, well-scaled products.

The [three failing regressions](evidence/issue-28/legacy-research/0028-factorized-extreme-red.txt) and
[repaired five-case output](evidence/issue-28/legacy-research/0028-factorized-focused-repair.txt) remain
separate. The first full-check attempt was interrupted after this independent
finding; its [output](evidence/issue-28/legacy-research/0028-factorized-check-ec522ed-interrupted.txt) prints
602 passed and one skipped but exits unsuccessfully through Make with error 130.
It is not a completed successful `make check`.

The standards review found no hard documented violations and one new nonblocking
heuristic: preallocation sampling duplicated the stochastic node's resource
constants. `ec39b46` shares its default parameter count and dimension-based working
storage calculation. The older profile-registry duplication remains a separate
known heuristic. Final scoped rechecks and full verification follow below.


## Author census and retained procedures

At `ec39b46`, a fresh author CPU census retains **192 valid / 192 attempted**
tasks, 64 for each topology family, seed 28 and task indices 0 through 63.
Each uses 16 rows, eight context rows and four observed features, cycling
regression/binary/three-class adapters. There are 1,739 factorized projections:
1,211 with inner dimension 1, 385 with 2, 70 with 3 and 73 with 4. These are
construction dimensions, not measured numerical ranks. Activation preference
modes occur 64 independent / 63 Dirichlet / 65 homogeneous times.

Overlapping valid flags are 61 disconnected targets, 27 inactive ReLU, 25
saturation and three zero-target-dispersion cases. No valid draw was removed;
no task was retried. The [summary and artifact hashes](evidence/issue-28/legacy-research/0028-factorized-census.json)
identify every original JSONL and successful saved group under the recorded local
output directory. Exogenous counters cover explicitly recorded stochastic nodes;
legacy Gaussian root nodes do not have that descriptor and are not counted there.
Each recipe is captured in memory before generation; JSONL and saved groups are
written after each family finishes. Timing and peak process memory are NOT_MEASURED; this is not a maximum-size or throughput qualification.

Reproduce at the recorded revision with a new destination:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/causal_topology_census.py \
  /tmp/r0-28-factorized-census-reproduction --profile causal-factorized-v1
```

The earlier `ec522ed` census also passed 192/192 but used records predating the
explicit exact-arithmetic budget. Its original output remains separately at
`/tmp/r0-28-factorized-census-ec522ed/`; it is not substituted for the repaired
recipe identities. Focused authored checks use:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_weights.py --tb=short
```

The [original and scoped standards reports](evidence/issue-28/legacy-research/0028-factorized-standards.md)
record no hard violations and resolve the new resource-accounting heuristic.
The standards reviewer did not run tests or independently certify numerical
correctness, performance or supported sizes. The profile-registry heuristic
remains nonblocking. Broad paper coverage and browser/Colab/GPU evidence remain
required and incomplete.


## Full repository verification

At `ec39b46`, complete `make check` passes **605 tests, 1 skipped**, with all
preflight/static gates passing. The [complete output](evidence/issue-28/legacy-research/0028-factorized-check-ec39b46.txt)
is retained verbatim. Command and environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

The skip is the existing GPU agreement check because CUDA is unavailable.
Smoke/eval-smoke remain NOT_AVAILABLE: T7/T12 train/eval CLI commands have not
landed. These are local CPU checks, not hosted CI, browser/Colab interactions,
predictive-performance evidence or full #28 acceptance.


## Final independent scoped verification

Independent spec/public-interface recheck **PASS** at `ec39b46`: all **33 check
groups** pass. Fourteen retained Fraction-arithmetic cases independently check
extreme, signed, zero, rectangular and subnormal products; nine nearby finite
boundary cases also pass. Exact-work admission accepts eight terms and rejects
seven for a 2×2×2 construction. Invalid budgets and genuinely nonrepresentable
final projections reject explicitly. This is independent rational arithmetic,
not a duplicate of the implementation's Decimal fallback.

Seventy-two sampled recipes all generate successfully. Their factor dimensions,
rank bounds, scaling and admitted parameter counts match independent calculations.
The observed scaled entry second moment is 0.995526 (expected 1); the factor second
moment is 1.010268 (expected 1). These are bounded sample diagnostics. Separate
RNG instrumentation verifies the uniform inner-dimension and standard-Gaussian
calls, and observes admission immediately before factor-array allocation. Injected
admission rejection prevents those allocations. This instrumented evidence is
separate from the arithmetic and finite-sample checks.

All three topology families pass explorer backend generation, selected-row trace,
save, reopen and replay. Numeric prefixes persist when row count changes; source
task IDs correctly change because row count belongs to the recipe identity.
Exact full-recipe/seed/task replay preserves identity.

All **252 legacy outcomes** exactly match base `5562ca0`: 250 successful cases and
two preserved `nonfinite square activation output` failures in the exogenous
forward profile, seed 28/task 50, regression and binary controls. All four previous
profiles, all topology families and all three task types are covered. Failed
outcomes are retained, not silently excluded from the denominator.

The [receipt](evidence/issue-28/legacy-research/0028-factorized-recheck-receipt.json) identifies commands,
archives, script hashes and actual source revision before/after execution. The
[check report](evidence/issue-28/legacy-research/0028-factorized-recheck.json) and [artifact manifest](evidence/issue-28/legacy-research/0028-factorized-recheck-artifacts.json)
link ordered JSON parts for checks, rational cases, sampled records, RNG events
and legacy outcomes. Arrays were split only to satisfy the per-file size policy;
concatenation reconstructs every original value exactly. Original complete
reports and raw per-case artifacts remain under
`/tmp/factorized-spec-ec39b46-a4/`. The manifest retains their original hashes.

The [portable runner](../scripts/verify_causal_weights_repair.py),
[case checks](../scripts/causal_weight_cases.py) and
[legacy recorder](../scripts/causal_weight_legacy.py) are byte-identical to the
independently executed final package. They archive both revisions, verify imported
source paths/hashes and require a new output directory. Reproduce from an unchanged
checkout using its selected environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_weights_repair.py \
  --repo . --expected-head HEAD \
  --base 5562ca0f2f5f9628c5386c02755dae9229b778e4 \
  --output /tmp/r0-28-factorized-independent-reproduction
```

Earlier verifier attempts incorrectly expected an unchanged source ID after a
row-count edit and omitted classification class counts in part of the legacy
matrix. Those were harness defects, not implementation findings or successful
legacy coverage. The corrected final run above uses valid controls and retains
actual numerical failures. Earlier attempt directories remain separate; the
original implementation REJECT is not overwritten. Publication adds only docs
and verification programs after `ec39b46`; package and test source are unchanged.
The full #28 goal remains active.
