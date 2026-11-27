# Issue 28: joint source activation and matrix profiles

This increment integrates the [source activation profiles](issue-28-source-causal-profiles.md)
and [recorded matrix preferences](issue-28-shared-matrix-preferences.md). It retains
their source references and declared r0 adaptations. The implementation is
independent; upstream code was not copied. This is one required integration toward
#28, not a complete published joint-prior reproduction or whole-issue acceptance.

## Public contract and selection scope

`with_source_matrices(base, prior=SourceActivationPrior(...),
matrix_prior=CausalMatrixPrior(...), seed=..., task_id=...)` returns a
`SourceMatrixCausalRecipe`. The base must be an ordinary sampled Gaussian-affine
recipe. Specialized recipes are rejected to preserve their provenance. Both priors
must agree on local/global/meta mode and shared/refresh categorical policy.

One source activation sampler owns selections for **both** transformed matrices
and node stages. Matrix transformations use the plain selector. Node stages use
the declared standard or plain selector. Named fixed-primitive preferences and
power exponents are therefore shared wherever their names overlap. In global
plain mode, every transformed matrix and active node stage uses the same retained
primitive, including any exponent. Standard and plain selector-slot preferences
have different names, while the fixed-primitive catalog has one shared name.

Sampling order is explicit: plan every non-root matrix in node order, selecting
its plain activation when needed; then select active node input/output stages in
node order and their standard-wrapper rescale draws; retain noise-order units;
admit the combined recipe; materialize matrix draws; fit independent ancestors.
Matrix family choices, vector hyperparameters and kernel-point choices use the
existing named matrix scopes. Numerical exponent, rescale and matrix scopes have
disjoint names. Categorical hyperparameter vectors retain fresh nested numerical
scopes, as documented in the preceding matrix checkpoint.

`matrix_activation_nodes` binds the ordered prefix of source selections to all
transformed matrices exactly once. Remaining source selections bind to every
active node stage. Matrix field bindings retain every categorical and numerical
decision, including child constructions and mixture component counts. Loading
checks both complete binding catalogs, all selected values, prior policies and
the fitted ancestral replay. No orphan or silently discarded selection is allowed.

The seed derives from `source-matrix-prior-v1`, both prior records, seed and task
id. Task rows/context and admission budgets are excluded from that identity.
Changing observations cannot refit the mechanism. Different profile names are
explicit exploration recipes, not bitwise upstream RNG reproductions.

## Calibration and admission

All five matrix families remain available: Gaussian, importance, factorized,
Laplace kernel and transformed. Transformed matrices have a nonrecursive simple
child, flattened plain activation, retained jitter and row normalization. The
matrix prior retains polynomial/uniform positive vectors, both point-cloud laws
and both covariance bases. Existing Gaussian roots and Gaussian additive structural
innovations remain the base for this integration. Broader exogenous integration
is still required; these APIs do not silently reinterpret a specialized base.

Standard node wrappers calibrate through the materialized matrix equations in
parent-before-child order. Root arithmetic retains its existing Gaussian path.
Matrix nodes compute bias plus activated-parent projection **before** adding
innovation for a before-output stage, matching their generation equation. An
after-output stage adds innovation only after the frozen output wrapper. Each
stage retains independent stream identity, sample digest, shifts and fitted
constants. Plain profiles do not allocate an ancestral calibration sample.

Metadata, matrix constructions, Gaussian stochastic descriptors, fitted wrappers
and their duplicated stored realizations all count toward the combined parameter
limit. Before matrix arrays or fitted tensors, admission includes the maximum
of task/calibration rows and the largest matrix-noise or fitted-stage channel
buffer. Limits remain conservative array estimates, not measured process RSS or
hard operating-system enforcement. Numerical and categorical preference limits
apply separately to each explicit scope; the causal limit admits their total.

## Exploration profiles and delivery

Eight profiles support forward, layered and redirection graphs:

| Profile | Selector | Mode | Categorical policy |
| --- | --- | --- | --- |
| causal-source-standard-matrices-v1 | Standard | Meta | Shared |
| causal-source-standard-refreshed-matrices-v1 | Standard | Meta | Refresh |
| causal-source-standard-local-matrices-v1 | Standard | Local | Shared |
| causal-source-standard-global-matrices-v1 | Standard | Global | Shared |
| causal-source-plain-matrices-v1 | Plain | Meta | Shared |
| causal-source-plain-refreshed-matrices-v1 | Plain | Meta | Refresh |
| causal-source-plain-local-matrices-v1 | Plain | Local | Shared |
| causal-source-plain-global-matrices-v1 | Plain | Global | Shared |

They retain exploration widths 1/2/4, input/output/both activation placement,
noise-after probability 0/0.5/1 and 256 independent calibration rows. These are
explicit r0 exploration settings. Valid disconnected, saturated or constant draws
are retained and flagged. Numerical failure is explicit; no retry or replacement
is introduced. Shared generation and saved explorer paths use the same recipe
decoder and sampler. Actual browser/Colab acceptance remains separate and pending.

## Verification state

The full `make check` run at `bb78d8174ebe322b7784027e425d16414525ede1`
passes **738 tests, one skipped**, all static gates. The skip is the existing CUDA
case on this CPU-only environment. [Full output](evidence/issue-28/legacy-research/0028-source-matrix-check-bb78d81.txt).

[Development evidence](evidence/issue-28/legacy-research/0028-source-matrix-development.json) retains RED
for the missing API and unsupported profiles, targeted GREEN, the original
unfiltered saved-workflow failure and its complete sampled recipe. Twenty-three
new backend identities save/reopen/replay successfully; the one exp-overflow case
is retained as an explicit failure assertion. Targeted tests cover all five
matrix families, both noise orders, calibration replay, row/context independence,
corrupted bindings and exact parameter/working-byte boundaries. These checks do
not substitute for the independent numerical evidence below.

[Standards review](evidence/issue-28/legacy-research/0028-source-matrix-standards.txt) finds no documented
violations and one advisory duplicated-stage-planning concern. The two paths
currently agree on retained costs; future wrapper changes must keep their planning
and accounting aligned. This is separate from mathematical acceptance.

## Independent spec and numerical verification

The [independent review](evidence/issue-28/legacy-research/0028-source-matrix-spec-review.txt) gives scoped
PASS with no implementation defect identified. Forty-five fixed cases independently
reconstruct all five matrix families, kernel points, flattened transformations,
ancestral fitting and query generation across mixed widths, placements, vector laws
and noise orders. Maximum error divided by `1 + abs(expected)` is **6.54e-11**,
within unchanged `rtol=atol=2e-9`. Absolute maximum is **6.10e-5** on large values;
these are distinct error measures. Row/context changes preserve mechanisms.

Thirty-three malformed-record/admission checks reject. Independent accounting
matches **2,460 parameters and 347,136 estimated working bytes** for its declared
fixture, with exact limits accepted and minus-one budgets rejected. Six plain
mode/policy variants and six saved regression/binary/multiclass workflows pass.
Twenty-four public profile/graph combinations round-trip; sixteen fixed global
joint seeds exercise four shared fixed-primitive overlaps and one shared power
exponent overlap. All twenty legacy profiles retain **61 byte-identical files**
against `b21f353`, including recipes, tasks, traces and their report.

The first NumPy oracle stopped after twelve successes: propagating NumPy ancestor
values with fitted Torch constants left roughly 1e-16 residuals at a fractional-
power zero, amplified to a 2.38e-6 output difference. Independent Torch equations,
without the implementation evaluator, reproduce all four affected calibration
sample digests exactly. The corrected NumPy oracle fits and propagates its own
constants during calibration, then uses frozen recipe constants for queries. The
same full grid and tolerances pass. A supplemental fixture initially requested
eight observed features from seven channels; the original failure is retained,
and the complete corrected two-feature fixture passes. Neither harness stop is
reported as an implementation rejection or silently omitted.

[Raw evidence](evidence/issue-28/legacy-research/0028-source-matrix-spec-artifacts.json) retains 1,294 files
in fourteen ordered ZIP byte parts, including original stopped runs, diagnoses,
corrected complete runs, inputs, recipes, saved workflows, exact script/import
hashes and custody. Concatenate parts without separators and verify the manifest
SHA-256 before extraction. The excluded baseline checkout is reproducible from its
Git revision. Five published `scripts/verify_source_matrix_*.py` programs are
byte-identical to final executed, Ruff-clean copies. Original programs also remain
in the archive; publication is supported by fresh final executions, not an assumed
formatting-only identity.

The numerical, contracts, compatibility and scopes programs take a fresh output
directory; run compatibility against base and candidate source trees separately.
The roundoff program takes the extracted evidence root and a fresh output directory.
Use the CPU/thread environment shown in the census reproduction below. Full source
checks and the author census are separate from the independent evidence. Actual
browser/Colab/GPU, supported-size and whole-issue acceptance remain pending.

## Unfiltered author census at bb78d81

All eight profiles attempted seed 28 and task ids 0..63 for forward, layered and
redirection graphs, cycling regression/binary/multiclass with 16 rows and four
observed features. **1,509 valid / 1,536 attempted** tasks are retained. All 27
failures occurred during generation; every attempt has a complete sampled recipe.
No sampling or calibration attempt failed, and no failed or degenerate draw was
replaced. Failures are explicit exponential or integer/ReLU/signed-power overflow.

| Profile suffix | Valid / attempted | Summary and artifacts |
| --- | --- | --- |
| standard | 192 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-standard.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-artifacts.json) |
| standard-refreshed | 191 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-refreshed.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-refreshed-artifacts.json) |
| standard-local | 192 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-local.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-local-artifacts.json) |
| standard-global | 191 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-global.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-standard-global-artifacts.json) |
| plain | 185 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-plain.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-artifacts.json) |
| plain-refreshed | 186 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-refreshed.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-refreshed-artifacts.json) |
| plain-local | 191 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-local.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-local-artifacts.json) |
| plain-global | 181 / 192 | [summary](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-global.json), [artifacts](evidence/issue-28/legacy-research/0028-source-matrix-census-plain-global-artifacts.json) |

Every header/recipe identity and complete failed record is published. Successful
raw recipes and task archives are explicitly LOCAL_ONLY, with hashes and
reproducible controls retained. Timing and process peak RSS are NOT_MEASURED.
The [occurrence counts](evidence/issue-28/legacy-research/0028-source-matrix-census-coverage.json) cover all
31 primitive kinds and all five matrix families across attempted recipes; these
are pooled occurrence counts, not per-profile exhaustive coverage or efficacy.

Run from a checkout of the stated source revision so the census Git receipt also
identifies the imported code.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/bb78d81/src /path/to/python scripts/source_causal_census.py \
  /new/output --profile causal-source-standard-matrices-v1
```

Repeat with each profile and a fresh output directory. Numerical correctness and
actual browser/Colab/GPU acceptance require their separate evidence.
