# Issue 28: source activation preferences on sampled causal graphs

This increment binds the [source selectors and frozen wrappers](issue-28-source-activations.md)
to sampled Gaussian-affine causal structures. Eight profiles support the three
existing graph families and the shared saved backend workflow. Matrix-construction
integration, remaining wrappers, wider mechanisms, typed states, interventions and
actual browser/Colab delivery remain required. This is not a complete published
joint-prior reproduction or completion of #28.

## Public sampling contract

`with_source_activations(base, prior=SourceActivationPrior(...), seed=..., task_id=...)`
accepts a base `gaussian-affine-v1` recipe with plain Gaussian causal nodes. It
returns `SourceCausalRecipe`. Already specialized activation/exogenous/matrix
recipes are rejected instead of silently discarding their sampling provenance.
Matrix integration remains a separate required extension.

`SourceActivationPrior` retains standard/plain selector, local/global/meta mode,
shared/refresh categorical policy, input/output/both placement, noise-after
probability and independent calibration row count (default 256). Output activation
can apply to roots; input activation only applies where parent channels exist.
Independent numerical `rr_scale` draws use the pinned source's log range [1,10]
and share a named Beta hyperprior, or a global value, according to the mode. The
numerical exponent and rescale scopes have disjoint names. Keeping their immutable
records separate does not replace a shared name with independent hyperparameters.

Every active stage binds exactly one retained primitive selection and, for the
standard selector, one rescale draw. Stage references cover all active stages and
all draws in node/input/output order. Loading checks complete coverage, selector,
primitive, rescale, preference policy and noise-order realization. Inactive stages
must be identity. The source records retain numerical/categorical preferences,
noise-order units, stage bindings and calibration stream identities.

The additional numerical/categorical `PreferenceLimits` arguments allow larger or
smaller selection budgets without changing the random stream identity. Each
numerical scope has its own limit; the categorical limit includes its nested
weight hyperpriors. Whole-recipe `CausalLimits` also admit their combined retained
cost. Metadata limits guard sampling before fitted tensors are allocated. A rejected
request is not automatically retried or replaced.

## Independent ancestral calibration

Primitive and rescale decisions are sampled first. Standard activations then fit
on a separate Gaussian ancestral realization, following parent-before-child order:

1. Fit an input wrapper on the already-fitted parents' concatenated channels.
2. Apply that frozen input wrapper and the recorded projection/bias.
3. Include structural innovation before or after the output stage according to
   the retained noise-order unit.
4. Fit the output wrapper on its actual pre-activation argument, then retain the
   resulting channel values for descendant calibration.

Streams are identified by `source-activation-calibration-v1`, calibration seed,
task id, node/channel or shift-stage keys. Stage identities and sample digests are
stored in each fitted activation. Fitting uses neither task rows nor query values
or labels. Changing task row/context counts leaves all mechanisms, preferences and
fitted constants unchanged. Plain selectors require no calibration sample.

`SourceCausalRecipe` loading independently regenerates the ancestral calibration
using retained primitive choices, rescale draws and the calibration stream. Every
fitted record must equal that replay, including means, inverse scales, shift rows,
digests and identity. Editing a constant without a corresponding coherent refit is
rejected. This check is CPU float64 and adds bounded replay cost during loading;
it is not a claim of cross-environment or GPU numerical acceptance.

The fit uses population statistics of its finite independent calibration sample.
It does not claim exact population unit variance for the generating distribution.
Genuine nonfinite sampling/calibration/evaluation results are explicit failures.
No clipping, label-based rejection or replacement of valid degenerate draws is
introduced. The existing whole-task normalization and moment rules remain separate.

Geometry is admitted before fitting tensors. Retained costs include fitted channel
constants, duplicated preference realizations, stage indices, noise units and
calibration identities' numeric fields. Conservative working admission includes
calibration row count and fitted-stage channel buffers; the generation card reports
the same calibration-aware admission. This is an array estimate, not measured RSS.

## Exploration profiles

Every profile supports forward, layered and redirection graphs. These retain the
existing exploration graph/width design, sampling widths from 1/2/4, placement
from input/output/both, and noise-after probability from 0/0.5/1. Calibration uses
256 independent rows. These exploration settings are explicit r0 adaptations.

| Profile | Selector | Mode | Categorical policy |
| --- | --- | --- | --- |
| causal-source-standard-v1 | Standard | Meta | Shared |
| causal-source-standard-refreshed-v1 | Standard | Meta | Refresh |
| causal-source-standard-local-v1 | Standard | Local | Shared scope |
| causal-source-standard-global-v1 | Standard | Global | Shared scope |
| causal-source-plain-v1 | Plain | Meta | Shared |
| causal-source-plain-refreshed-v1 | Plain | Meta | Refresh |
| causal-source-plain-local-v1 | Plain | Local | Shared scope |
| causal-source-plain-global-v1 | Plain | Global | Shared scope |

Local/global modes do not use a categorical weight distribution, so its refresh
choice would not add a distinct law. Direct APIs expose all prior options, noise
probabilities in [0,1] and configurable positive calibration row counts subject to
resource admission. No Cartesian-product maximum or bulk throughput is claimed.

## Current evidence and pending qualification

Development tests use public sampling, JSON decoding, generation/trace and saved
backend seams. A four-node mixed-width example verifies seven stage bindings,
standard rescale coverage, row-count independence, complete admission reporting,
replay and corruption rejection for both selectors.

The initial unfiltered 24 profile/topology save attempts (seed 28, task id 10,
16 rows, eight default features) produced **22 successful workflows and two explicit
exponential-overflow failures**, both plain/refreshed (forward and layered).
The same failed identities remain in the qualified check; their exact failure
messages and full sampled recipes are retained. No failed draw was replaced.
The other 22 workflows save, reopen, regenerate and inspect their original rows.
This is backend evidence, not actual browser or Colab evidence.

Development logs and the two initial failed recipes are retained in the
[development manifest](research/0028-source-causal-development.json), separately
from frozen-source verification below. The full #28 goal remains active.


## Frozen verification checkpoint

At `c4081e1`, [full repository checks](research/0028-source-causal-check-c4081e1.txt)
pass **679 tests, one skipped**, including all static gates. The skip is the existing
CUDA case on this CPU-only environment. Smoke/eval-smoke remain NOT_AVAILABLE:
this checkout exposes no train/eval CLI. Publication preserves these source/test
bytes; later documentation, census and portable verifier programs are separate.

[Independent spec review](research/0028-source-causal-spec-review.txt) passes 27
NumPy ancestral-calibration/generation cases: three modes, three placements and
three noise probabilities, mixed widths 2/1/3/1, 32 calibration rows and 16 task
rows. It checks stage statistics, shifts/indices, identities, parent propagation
and generated values through public traces. Maximum error divided by
`1 + abs(expected)` is 4.164e-13 within the declared `rtol=atol=2e-9` tolerance.
The maximum absolute error is 23,552 on very large exponential intermediates;
this is not an absolute-small-error claim.

Twenty-four corruption/admission cases reject. Independent integer accounting
matches **415 parameters and 85,376 estimated working bytes** for its declared
fixture: exact boundaries accept and minus-one budgets reject. Existing selection
record counts are reused; the new stage/rescale integration costs are independently
enumerated. Larger selection budgets and changes to task rows/context preserve
mechanisms. Six plain policy records decode, and six standard/plain by
regression/binary/multiclass explorer workflows save, open, replay and trace.
Twelve old-profile fixtures produce **37 byte-identical files** against `de9262f`.

[Raw spec evidence](research/0028-source-causal-spec-artifacts.json) retains original
and final executions, exact inputs, fitted recipes, saved workflows, source/import
hashes, custody and historical harness errors in six ordered ZIP byte parts.
Concatenate without separators and verify the manifest SHA-256 before extraction.
The baseline checkout is excluded and can be reconstructed from its Git revision.
The three published `scripts/verify_source_causal_*.py` programs are byte-identical
to final executed publication copies. Their cleanup and bounded rerun are described
in the [publication qualifier](research/0028-source-causal-publication.txt); original
programs, outputs and PASS report remain retained. No formatting-only identity is
claimed. The wrong-keyword harness attempt and unused NumPy-branch overflow warning
remain separate from product findings.

[Standards review](research/0028-source-causal-standards.txt) finds no documented
violations and one low-severity possible duplication heuristic in metadata cost
accounting. It does not replace numerical verification.

## Unfiltered author census

All eight profiles attempted seed 28 and task ids 0..63 in every topology, cycling
regression/binary/multiclass, with 16 rows and four observed features. **1,507 valid
/ 1,536 attempted** tasks are retained. All 29 failures occurred during generation;
every attempted task has its complete sampled recipe. No calibration attempt failed,
and no failed or degenerate task was replaced.

| Profile suffix | Valid / attempted | Summary and artifact manifest |
| --- | --- | --- |
| standard | 191 / 192 | [summary](research/0028-source-causal-census-standard.json), [artifacts](research/0028-source-causal-census-standard-artifacts.json) |
| standard-refreshed | 192 / 192 | [summary](research/0028-source-causal-census-standard-refreshed.json), [artifacts](research/0028-source-causal-census-standard-refreshed-artifacts.json) |
| standard-local | 192 / 192 | [summary](research/0028-source-causal-census-standard-local.json), [artifacts](research/0028-source-causal-census-standard-local-artifacts.json) |
| standard-global | 188 / 192 | [summary](research/0028-source-causal-census-standard-global.json), [artifacts](research/0028-source-causal-census-standard-global-artifacts.json) |
| plain | 184 / 192 | [summary](research/0028-source-causal-census-plain.json), [artifacts](research/0028-source-causal-census-plain-artifacts.json) |
| plain-refreshed | 185 / 192 | [summary](research/0028-source-causal-census-plain-refreshed.json), [artifacts](research/0028-source-causal-census-plain-refreshed-artifacts.json) |
| plain-local | 190 / 192 | [summary](research/0028-source-causal-census-plain-local.json), [artifacts](research/0028-source-causal-census-plain-local-artifacts.json) |
| plain-global | 185 / 192 | [summary](research/0028-source-causal-census-plain-global.json), [artifacts](research/0028-source-causal-census-plain-global-artifacts.json) |

Failures are explicit exponential or integer/ReLU/signed-power overflow. The pooled
attempted recipes cover all **31 primitive kinds** (27 fixed plus four power
families); this does not say each profile individually covers all kinds. Valid
flagged tasks remain in the denominator. Every header/recipe identity and every
complete failed record is published. Successful raw recipes and task archives are
explicitly LOCAL_ONLY in the manifests; their hashes and reproducible controls are
retained. Timing and peak RSS are NOT_MEASURED for the census.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/c4081e1/src /path/to/python scripts/source_causal_census.py \
  /new/output --profile causal-source-standard-v1
```

Repeat with each desired profile and a new output directory. Independent verifier
programs take one fresh output-directory argument with the same environment; run
compatibility separately against the base and candidate source trees.
This establishes bounded source-activation integration, not matrix integration,
all-paper completion, browser/Colab/GPU acceptance, measured supported-size limits
or predictive superiority.
