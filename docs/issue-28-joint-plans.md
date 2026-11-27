# Joint activation, converter and weight plans

This implements typed consumers of the [shared context](issue-28-joint-sampling.md)
from the [integration design](research/0028-pipeline-sampling-integration.md).
The [accepted issue scope](specs/0028-causal-network-generators.md) remains broader:
matrix/point/root plans, aggregate semantic ownership, complete ancestral fitting,
the jointly sampled graph recipe and workbench delivery remain required.

## Plans and semantic bindings

The public `priors` entry point exposes `JointActivationPlan`,
`JointConverterPlan`, `JointNumericalConverterPrior`, `JointWeightPlan`, their
`sample_joint_*` functions and checked `joint_*_plan_from_dict` decoders.
Every plan records its full operation path. `JointSamplingContext.operation`
provides that path through the same closed/active-child guards as sampling.

`plan.validate(decisions)` checks exact source requests, branch roles/order,
ownership paths and realized values. Activation and weight operations are leaves;
activation converters own their mode plus an `activation` child. Extra/missing
events under a plan's subtree reject. Unrelated operations elsewhere in the ledger
are permitted so the plans can compose. Decoders require a frozen decision record
and validate bindings before returning a plan. Operation paths and activation
parameter pairs must be arrays; dictionaries/strings are not coerced into them.

This per-plan binding contract does not prove a complete graph's destination map
or unique coverage by a collection of plans. A future aggregate registry must
validate all destinations and reject duplicate plan operations, including uniform
weights whose deterministic configuration has no named decision events. No fake
forced-identity draw is inserted for an event-free weight plan.

## Source activation choices

`sample_joint_activation(context, name, selector="standard")` retains the existing
four standard/eight plain fixed-family slots, all 27 fixed primitives, both common
power families and the plain-only reciprocal/integer powers. It uses the existing
source request builders; repeated slots retain their separate categorical masses.
Plain order is selector, then primitive/exponent. Standard order is selector,
`rr_scale`, then primitive/exponent, matching the pinned source's lazy wrapper
order: standardize, rescale, chosen primitive, standardize.
[Pinned activation source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L8-L29).

Names such as `acts_torch` and power exponents therefore share the outer
preferences across matrix/node/converter operations. `act_type` and
`act_type_plain` retain distinct catalogs. `rr_scale` joins the same outer scope,
instead of a separate rescale sampler. Standard plans require a finite scale in
[1,10]; plain plans have none. Choices freeze before independent fitting, preserving
the accepted adaptation rather than reproducing source parameter redraws.

## Numerical converter choices

`sample_joint_converter(context, name, channel=..., feature_group=..., prior=...)`
uses `num_conv_mode` with ordered source labels `id`, `kumar`, `act`.
`JointNumericalConverterPrior` has independent Kumaraswamy/activation switches,
`disallow_target_warping` and all three existing converter variants. Defaults
enable both warp families for broad exploration and use `readout-only`; these are
declared exploration defaults, not the pinned configuration's defaults.

Only the chosen branch draws parameters: Kumaraswamy owns log requests
`kumaraswamy_a` and `kumaraswamy_b` in [0.2,5]; activation owns a nested standard
activation plan; ordinary identity owns only its selector. Exact group `y` with
target warping disabled instead owns one
`forced_identity(reason="target-warp-disabled")` and consumes no RNG. Groups `Y`
or `y-extra` are not targets under this rule. The full task-wide catalog remains
unchanged, avoiding a singleton-catalog cache collision. This forced branch is an
explicit consumption adaptation, not a claim of reproducing the source bug.
[Pinned converter source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py#L46-L73),
[target-group source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L23-L34).

## Weight batches

`sample_joint_weights(context, name, width=..., batch=1, family="polynomial",
scalarization="scalarized-weight-batches-v1")` returns one batch plan. Uniform
plans contain no polynomial parameters or named draws. Polynomial plans contain
one decay/scale per row and share the original outer names across consumers;
categorical nested weight hyperpriors remain separate.

Both policies preserve all-decays-before-all-scales order:

- `scalarized-weight-batches-v1` uses scalar requests and roles `decay:0` ... then
  `scale:0` ... . Global quantile policy ties row parameters and remaps bounds for
  different widths; this is an explicit source-array adaptation.
- `source-weight-batches-v1` retains two requests with shape `(batch,)`, roles
  `decay` and `scale`. Meta/local share named preferences across batch sizes.
  Global use requires the context's `strict-contract-v1`: cached arrays retain
  distinct row units, while incompatible shapes or bounds reject atomically.
  Source's unchecked incompatible-array reuse is not silently accepted.

`materialize_joint_weights(plan, decisions, rng=..., limits=...)` always returns a
tuple of vectors, including batch one. It draws the full Gaussian batch before
per-row permutations, or the full uniform batch. Existing frozen weight records
provide arithmetic and RMS-one polynomial normalization. Named choices and raw
realizations use separately injected streams; explicit NumPy/float64 execution
does not claim source NumPy/Torch RNG parity.
[Pinned weights source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py#L7-L56).

## Fitting and admission

`fit_joint_activation(plan, decisions, calibration, rng=..., calibration_id=...,
limits=...)` validates bindings and atomically fits a standard wrapper using its
frozen primitive/scale. Plain plans return their primitive without fitting and
may pass `None` as calibration. `fit_joint_converter` accepts a finite nonempty
one-channel calibration sample and produces the existing `NumericalConverter`,
including variant/channel. Identity fits no constants; the other branches reuse
existing frozen Kumaraswamy or standard-activation arithmetic.

These helpers stage calibration/materialization RNG and commit only after
successful construction. They never redraw the planned mechanism. Fit helpers
accept no task rows, labels or query objects, but caller-provided identity/sample
alone cannot prove independence or actual ancestor provenance. Population moment
certification and fitting against finalized parent states remain responsibilities
of the future graph fitter. A finite empirical sample does not prove moments.

Choice budgets include raw batch cardinality before parameter tuples and exact
native preference costs incrementally. Source array requests inherit the native
100,000-value per-request shape cap. Plan decoding checks available numerical
draw/value counts before copying batch tuples. Materialization admits the whole
batch before arrays. Causal limits on fitting cover the output converter including
its channel, or standard activation; vector limits cover the output batch. Plan
`parameter_count` exposes retained realized metadata separately. These local gates
do not replace aggregate admission of all plans, ledger, graph buffers, matrices
and fitted constants. Estimated admission is not measured RSS.

## Development evidence

Ten new public cases plus 16 shared-context cases pass the focused command:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_joint_activation_plans.py \
  tests/priors/test_joint_converter_plans.py tests/priors/test_joint_weight_plans.py \
  tests/priors/test_joint_sampling.py
```

Environment: existing Python 3.12.3, NumPy 2.5.5, torch 2.15.0+cpu; no install or
cloud execution. Local logs are retained under
`docs/evidence/issue-28/joint-plans/`. Cases cover shared primitive/scale identity,
independent standard-wrapper arithmetic, strict pair decoding, exact target
restriction, all three converter generation/JSON/trace variants, weight widths,
source ordering, materialization and strict global arrays. A development decoder
coercion and typecheck issue were corrected; their failed logs remain retained.

Frozen source: `be3643989df7349a295a62e242db9522a03cda5d`; comparison base:
`800af3e4cd0888fcea5232df452490b65a7a3f0f`.
[Standards review](evidence/issue-28/legacy-research/0028-joint-plans-standards-be36439.txt) found no
documented-rule breaches and two nonblocking P3 heuristics: repeated operation-path
decoding and the converter-input helper's `_target` name. No source changes were
made for these suggestions during the review freeze.

Full `make check` at that source passed **835 tests, one skipped**, including all
static gates; pytest took 449.83 seconds. The command uses the environment above
with `make check` replacing the focused test invocation.
[Raw full output](evidence/issue-28/legacy-research/0028-joint-plans-check-be36439.txt) and
[author archive metadata](evidence/issue-28/legacy-research/0028-joint-plans-development.json) retain
23 development/check/help/report files. The archive includes every file's length
and SHA256 in `FILES.json`; all entries were read back and verified after packaging.
Intermediate RED/GREEN and tooling failures remain separate from frozen-source
acceptance. Raw logs are preserved byte-for-byte.
[Publication gates](evidence/issue-28/legacy-research/0028-joint-plans-publication-gates.txt) pass checkout,
hooks, tracked-file and artifact-size checks, formatting, lint and file-token caps.
Source/tests remain identical to the frozen source; these gates cover the added
documentation and verifier programs without repeating the unchanged full suite.

Independent specification review: **PASS_COMPONENT** with 2,161 assertions,
including 196 expected rejections. All 27 fixed primitives and power branches,
converter variants/switches, both batch policies, arithmetic, generation/JSON/trace,
binding corruption, exact limits and late-failure rollback are covered. All 162
legacy comparisons match the base byte-for-byte: 159 completed results and three
unchanged numerical failures. Every imported module is fingerprinted per program.
[Report](evidence/issue-28/legacy-research/0028-joint-plans-spec-report.txt),
[totals](evidence/issue-28/legacy-research/0028-joint-plans-spec-totals.json),
[exact reproduction and environment](evidence/issue-28/legacy-research/0028-joint-plans-spec-reproduction.txt).

Three verifier runtime mistakes (catalog transcription, local-preference semantics
and a decoder-name typo), intermediate lint failures and their exact programs and
outputs remain retained. None is a product REJECT. The six final programs under
`scripts/verify_joint_plans/` are byte-identical to the executed, formatted versions.
The legacy fixture program extends the previous verifier with six context cases;
the independent arithmetic checks do not call production helpers as their oracle.

The current [CLI help](evidence/issue-28/legacy-research/0028-joint-plans-cli-help.txt) exposes replay, oracle and
diagnose-priors; train/eval commands are absent, so the conditional smoke targets
remain NOT_AVAILABLE. An initial `python -m r0` help attempt failed because this
package has no `__main__`; the corrected console-entrypoint output is retained.
The complete matrix design is
[recorded separately](research/0028-joint-matrix-plans.md); its proposed oracles
remain NOT_RUN. No full issue, hosted/browser, GPU, scale or predictive claim is
made by this component.

## Retained reproduction

[Archive metadata](evidence/issue-28/legacy-research/0028-joint-plans-spec-artifacts.json) pins both parts,
the complete archive and the original 1,174-entry manifest. The archive retains
841 non-source files, with 578 duplicate paths restored through a map. Source and
base tarballs/exports are reconstructed from pinned Git objects and checked against
their original byte hashes. Restore to a new directory from this repository:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-joint-plans-spec-artifacts.json \
  /tmp/joint-plans-recovered --repo "$PWD"
```

This recovery was executed successfully for every original manifest entry;
[receipt](evidence/issue-28/legacy-research/0028-joint-plans-spec-recovery.json). Recovery verifies retained
artifacts; it does not count as fresh behavioral execution. For a fresh run, follow
the restored `REPRODUCE.txt`, copying programs and exports into a new replay
directory before running `RUN.sh`; keep retained outputs intact. Existing local
Python/NumPy/Torch are reused deliberately. Cached source excerpts are retained
with hashes, without a fresh upstream provenance claim.
