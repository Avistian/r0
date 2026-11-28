# Issue 28: recorded importance-weighted projections

This increment adds authored and sampled positive importance vectors and Gaussian
projections. It is partial issue 28 delivery; the scope ledger remains authoritative.
The prior factorized checkpoint is `977461c276c722592cac7bdd271d9250b93db972`.

## Frozen mathematical contract

`UniformWeightVector` retains an explicit nonempty tuple of draws in `[0,1)`;
its realized values are those draws. Zero is allowed, consistent with a uniform
floating-point generator's endpoint. It does not normalize the vector itself.

`PolynomialWeightVector` retains finite Gaussian realizations `z`, nonnegative
finite decay `d` and log-noise scale `s`, and a permutation (empty means identity).
For zero-based index `i`, compute `a_i = -d*log(i+1) + s*z_i`, permute the scores,
and return `sqrt(n)*exp(a)/sqrt(sum(exp(2*a)))`. This is mathematically softmax
followed by RMS normalization; the vector has squared norm `n`. Realized draws
are replay inputs, not assertions that an individual tuple passed a normality test.

The implementation centers scores before exponentiation. It computes products,
subtractions and centering using Decimal arithmetic with enough precision for
exact binary64 products; `log(i+1)` itself is a binary64 approximation. This avoids
losing small decay differences under a huge common perturbation. The declared
`exact_scores_limit` (default 1,000,000) bounds the number of scores before Decimal
work, and is resource control rather than a model parameter. Underflow in final
float64 values can produce zero. Epsilon-zero matrix normalization centers over
active Gaussian coordinates to retain a direction even when those weights alone
would underflow. Internal negative infinity for a zero weight is not serialized.

`ImportanceWeights` retains a finite rectangular Gaussian realization `G`, one
matching importance vector per row, and finite nonnegative `epsilon` (default
`1e-6`). For each row `r`, the projection is
`W_r = (G_r * v_r) / (epsilon + ||G_r * v_r||_2)`.
An all-zero numerator gives an all-zero row, including when epsilon is zero.
Scaled evaluation avoids overflow in the norm; extreme products use logarithms.
These are row-normalized projections, not orthogonal or singular-value-certified
matrices. Biases, activations, exogenous defaults, target controls and shared
causal generation semantics remain as recorded elsewhere.

A `MatrixCausalNode` stores both the realized projection and its construction;
recipe admission precedes construction materialization and rejects disagreement.
Parameter accounting adds all Gaussian entries, vector draws, polynomial decay
and scale, explicit permutation indices, and epsilon, as well as the existing
node and dense-projection parameters. Construction records travel through JSON,
row traces, saved archives, reopen and replay. A sampled factorized record must
contain actual factorized projections; importance provenance has its own guard.

## Source and local sampling choices

Primary references are pinned TabICLv2 graph-lib
[`_weights.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py),
[`_matrix.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py),
and [`_base.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py).
The vector decay/log-noise/shuffle law and Gaussian product with epsilon-plus-norm
follow those mathematical definitions. The implementation is independent code.

`causal-importance-v1` / `activation-importance-v1` is an r0 exploration profile:

- Forward, layered and predecessor-redirection graph families; widths 1, 2, 4;
  existing 26-choice activation preference law, including its retained modes.
- Each nonroot projection independently chooses uniform or polynomial vectors
  with probability one half. Each row has independent vector draws. Gaussian
  matrix entries are independent standard normals. Roots retain their existing
  activated Gaussian form; nonroots use default Gaussian exogenous/constant noise.
- For polynomial width `n`, decay lies between `0.1/log(1+n)` and 6. Draw local
  `u ~ U[0,1)`, transform `clip(1.2*u-.1,0,1)`, then interpolate logarithmically.
  Under the ideal continuous law this gives endpoint masses 1/12 each and interior
  mass 5/6. Log-noise scale is locally log-uniform from `1e-4` to 10. Each row gets
  its own standard-normal tuple and uniform permutation. Epsilon is `1e-6`.
- Admission accounts for retained arrays before allocating Gaussian/vector arrays.
  Sampling, row execution and saved replay do not retry or discard failed draws.

The one-half family mix, graph/width choices, local parameter independence and
stable float64 arithmetic are explicit adaptations. This does **not** reproduce
source GlobalSampler shared meta/global numerical preferences, recursive matrix
mixtures, kernel/transformed matrices, jitter wrappers, masks, full networks or
joint source priors. These remain required issue 28 work, not unsupported options
silently treated as complete. The old five causal profile streams must remain
unchanged, including their failures.

## Verification status

Implementation head `fc0265b7706112a0b674a2c85078f177fb265b83`:
- Uniform authored projection: RED `e5b4be7`, GREEN `c13a2ed`.
- Factorized family provenance: RED `f36065f`, GREEN `ff0a09f`.
- Polynomial laws, permutations and huge common offset: RED `879faea`, GREEN `534b3cb`.
- Saved explorer workflow on three graph families: RED `671a7e5`, GREEN `fc0265b`.
- Repository check and independent scoped review: PASS; evidence below.
- Publication adds documentation and verification programs; package/test source stays unchanged.

These are CPU software and retained-example checks. Scientific efficacy, speed,
CUDA behavior and complete browser/Colab acceptance are NOT_MEASURED/NOT_RUN.


## Retained verification and reproduction

`make check` passes **613 tests, 1 skipped** with all static gates passing.
It began at `fc0265b`; only the documentation commit `e6d8faa` landed while it ran.
Package and test bytes are identical at those revisions. The
[complete log](evidence/issue-28/legacy-research/0028-importance-check-e6d8faa.txt) and
[revision/context receipt](evidence/issue-28/legacy-research/0028-importance-check-context.json) retain that
qualification. The existing GPU agreement test is skipped because CUDA is absent;
T7/T12 train/eval CLI commands remain unavailable, so smoke/eval-smoke are NOT_AVAILABLE.

The [independent spec review](evidence/issue-28/legacy-research/0028-importance-spec-review.md) accepts only
this increment: **2,134 numerical checks**, 384 sampled recipes, four actual saved
backend sessions, and all **315 legacy outcomes**. The numerical oracle evaluates
Decimal exponential/square-root arithmetic independently of the implementation's
float64 materialization, including huge common offsets, product overflow,
subnormal inputs, active-support recovery, zero rows, invalid records and
construction disagreement. Legacy outputs match exactly: 313 successes and two
retained failures across the five older profiles. The sampler's finite-sample
moments and endpoint frequencies are diagnostics, not proof of a distribution.

Resource instrumentation matches the retained Gaussian arrays to allocation events
and observes full retained-array admission before all 22 matched allocations in
one declared forward example. Vector allocation ordering is source-reviewed.
Four backend cases cover forward/layered/redirection at seed 28 and an additional
explicit layered seed 0. The seed-28 layered graph has no nonroot projection and
remains valid. An initial oracle incorrectly required a nonempty projection trace;
that [harness error](evidence/issue-28/legacy-research/0028-importance-harness-error.md) is retained separately.
A second passing run verifies final import formatting; neither was a product rejection.

[Standards review](evidence/issue-28/legacy-research/0028-importance-standards.md) found no hard violations.
The existing repeated-profile-registry heuristic remains nonblocking.

The [receipt](evidence/issue-28/legacy-research/0028-importance-receipt.json) records exact revisions,
immutable archive hashes, interpreter, commands, program hashes and outcomes.
[Numeric details](evidence/issue-28/legacy-research/0028-importance-numeric.json),
[sampling diagnostics](evidence/issue-28/legacy-research/0028-importance-sampled-laws.json), and the
[ordered artifact manifest](evidence/issue-28/legacy-research/0028-importance-artifacts.json) preserve the
reported checks, resource events and legacy outcomes. Split arrays reconstruct the
original values exactly. Full raw records remain at
`/tmp/importance-spec-e6d8faa/portable-run-02/`.

The four published programs are byte-identical to that final independent run:
[runner](../scripts/verify_causal_importance.py),
[numeric oracle](../scripts/causal_importance_numeric.py),
[sampling/workflow checks](../scripts/causal_importance_cases.py), and
[legacy recorder](../scripts/causal_importance_legacy.py).
From a checkout containing both pinned commits, using the selected environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_importance.py \
  /tmp/r0-28-importance-independent-reproduction --repo .
```

The runner defaults to candidate `e6d8faa` and base `977461c`, archives both, and
requires a new output directory. Override `--candidate` and `--base` deliberately
when checking other revisions. This is retained CPU/backend evidence, not hosted
CI, browser, Colab, GPU, supported-size measurements or scientific acceptance.

## Unfiltered author census

The fixed seed-28 census attempted 64 recipes for each of three graph families,
cycling regression/binary/multiclass controls: **191 valid / 192 attempted**.
Forward task 47 (multiclass) failed with nonfinite integer-power activation output;
it was not retried, replaced or removed. The
[summary and raw-artifact hashes](evidence/issue-28/legacy-research/0028-importance-census.json) include its
full recipe as a header plus ordered node parts, with a verified reconstruction
hash. Raw JSONL and saved groups remain at `/tmp/r0-28-importance-census-e6d8faa/`.
Recipes are captured in memory before generation; JSONL is written only after a
family finishes, not durably before each attempt.

Across all 192 retained recipes there are 1,562 importance projections with 3,726 polynomial
and 3,899 uniform row vectors. Counts include the failed recipe. Among valid tasks,
75 have disconnected-target flags, 19 inactive-ReLU flags and 31 saturation flags;
these overlap and are not filtering criteria. Timing and peak memory are NOT_MEASURED.
Reproduce in a new output directory:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/causal_topology_census.py \
  /tmp/r0-28-importance-census-reproduction --profile causal-importance-v1
```
