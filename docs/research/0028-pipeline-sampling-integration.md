# Issue 28: joint source sampling and ancestral pipeline fitting

Design only. Read-only preflight 2026-11-27: `/tmp/r0-28` at
`962f0ada62a4eda3e71c45fa45bab0c4f66eb8fe`, pre-existing untracked `.venv`.
Inspected cached `/tmp/node-pipeline-sampling-research` files identified by
[prior audit][laws] as TabICL `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`.
No upstream execution, downloads, tests or source edits. Proposed oracles:
NOT_RUN. Local links refer to that frozen head; upstream links are pinned.
Authority: accepted decisions 11–13,18–19. [spec]

## Recommendation and boundary

Add a new recipe family: one outer numerical/categorical context, ordered
bindings, and one ancestral fit pass. Share preferences across matrices,
activations, weights, converters, noise and importance; retain nested categorical
scopes. Freeze mechanisms before independent fitting, a decision-12 adaptation.
[spec] [B] [N]

`with_source_matrices(...)` followed by independent postprocessing is insufficient:
its fitter already calibrated wrappers against ancestors without those pipelines.
Sharing its RNG does not share caches. These are code deductions, not executed
failures. The actual class is `SourceMatrixCausalRecipe`; no
`SharedMatrixCausalRecipe` symbol exists at this head. Add a separate
`SourcePipelineCausalRecipe`; preserve old families, serialized bytes, seed behavior
and validators. [joint] [fit] [sampler] [matrix] [source] [authored]

## What really shares in the pinned source

Derived `PriorComponent` contexts retain the same sampler. Meta shares
hyperparameters, global realizations, local no stochastic preferences; unnamed
Gaussian draws/permutations stay fresh. `SimpleRandomWeights(Context())` inside
categorical meta sampling deliberately owns a fresh nested sampler. [B] [W]

| Outer name | Consumers/constraints |
| --- | --- |
| `simple_random_weights_decay_rate`, `simple_random_weights_std_scale` | Node channels, matrix row/singular weights, mixture weights/factors, covariance roots. Width-dependent decay bounds and different batch shapes; excludes categorical nested weights. [W] [M] [P] [N] |
| `act_type`, `act_type_plain` | Distinct standard/plain catalogs, respectively four/eight duplicate fixed-family slots; preserve slot identities. [A] |
| `acts_torch`, `relu_random_power_exponent`, `random_power_exponent` | Shared across matrix plain and mechanism/converter standard activations. Negative/integer-power names are plain-only. [A] [M] [C] |
| `rr_scale` | Standard wrappers including converters share scale preference; shifts use unnamed per-channel row choices from their own standardized input. [A] [C] |
| `num_conv_mode`, `kumaraswamy_a`, `kumaraswamy_b` | Converter choice and separately named shapes; shapes/activation draws occur only on selected branches. [C] [A] |
| `gaussian_noise_sigma`, `node_importance` | Separate shared scalar names; importance multiplies innovation too. [N] |
| `random_matrix_type`, `simple_random_matrix_type` | Distinct catalogs; Gaussian-only configuration changes the former catalog and must be task-wide. [M] |
| `random_base_points`, `random_covariance_base_points`, `random_gmp_n_centers`, `random_kernel_matrix_dist_factor` | Root/point selectors and kernel parameters; reached weight consumers join the same outer scope. [P] [M] |

Meta numerical draws share location U and concentration H per name, then draw
Beta(UH,(1-U)H) with each request's own shape/bounds. Different widths can share
preferences without equal decay realizations. Global returns the first array
without checking shape/bounds; a matrix batch may be incompatible with the later
node `(1,)` request. Do not hide this by splitting names. [B] [W] [M]

Categorical `shared` models corrected cached-weight intent; `refresh` models the
historical cache-key error for ordinary calls. Source combines ambient NumPy/Torch
randomness with its sampler RNG, so injected RNG is not bitwise parity. [B]
[categories]

**Catalog collision:** unrestricted converters and restricted targets request
`num_conv_mode` with different catalogs, the latter `[id]`. Corrected meta can
reuse wrong-length probabilities; global can reuse an out-of-range index. Static
deduction, NOT_EXECUTED. Existing r0 rejects catalog changes even in `refresh`.
Recommend a task-wide unrestricted catalog plus explicit
`forced_identity(reason="target-warp-disabled")` for exact feature group `y`,
with no selector/shape/activation draws. Version this draw-consumption adaptation;
do not rename the target request or silently truncate probabilities. [C] [N]
[catrecords]

## Concrete records and migration seams

Proposed APIs, not implemented records. Reuse immutable primitives/transforms
where contracts match; preserve the existing strict ownership model. [spec]
[sampler] [bindings] [pipeline]

```text
SourceJointPrior(version, source_commit, mode, categorical_policy, global_policy,
  matrix/activation catalogs, vector_family, root_profile, width_policy,
  pipeline switches, converter_variant, target_policy, calibration_rows, limits)
SourceJointDecisions(version, numerical: ParameterPreferences,
  categorical: CategoricalPreferences, events, operations, bindings, adaptations)
DecisionEvent(ordinal, operation_id, kind, draw_index?, deterministic_reason?)
JointBinding(operation_id, role, destination_path, operation, element?)
ActivationPlan(selector, slot_event, primitive_event? | exponent_event?, rescale?)
VectorPlan(family, width, batch, decay_events, scale_events, realization_record)
ConverterPlan(node, channel, feature_group, mode_event? | forced_reason,
  variant, a_event?, b_event?, activation_plan?)
NodePlan(width, mechanism, root?, channel_vector?, standardize, l2,
  converters, sigma_event?, importance_event?)
CalibrationPlan(version, seed, task_id, rows, root/innovation/shift_streams)
FitReceipt(node, stage, input_digest, constants_digest,
  ancestor_state_digest, calibration_identity, rows)
SourcePipelineCausalRecipe(CausalRecipe,
  construction="source-joint-pipelines-v1", prior, decisions,
  calibration_plan, fit_receipts, pipelines)
```

Use tagged operation records: ledger owns draws, operations own references,
bindings map outputs to fields. One sigma broadcasts to channels without invented
extra draws. Validate names/shapes/transformations/destinations. [bindings] [numeric]

API: `JointSamplingContext.transaction()` stages `plan_source_graph`, complete
admission, materialization, `fit_joint_ancestral`, validation/freeze, then commit.
No task rows or labels enter the fitter. [spec] [fit]

Extract context-aware activation/matrix operations returning plans and global
event references. Resolve current transaction samplers, avoiding stale references
when forks replace them. Keep legacy compact-record adapters. [sampler] [planner]

Do not pass the aggregate ledger into old validators: `SourceActivationSelections`
requires contiguous full selector/primitive/exponent consumption and strict policy;
`validate_matrix_preferences` requires every draw to belong to a matrix field;
`SourceCausalRecipe` assumes separate rescales, exact node-stage ordering and the
old fitter. Reuse request/field-contract builders, with new aggregate ownership
validation. [sampler] [bindings] [source]

New matrix/root adapters need source-name factories: old
`kernel_covariance_base`, `kernel_mixture_components`, `kernel_distance_scale`
correspond to source `random_covariance_base_points`, `random_gmp_n_centers`,
`random_kernel_matrix_dist_factor`. Old `kernel_point_law` has no counterpart in
pinned RandomKernelMatrix, which directly uses Gaussian-mixture points. Preserve
old names as old adaptations; do not mistake them for complete source sharing.
[matrixrequests] [M] [P]

### Coherent global policy

Recommend default `meta`: one Beta hyperprior per name across differing widths.
For global exploration declare `scalar-quantile-v1`: scalarize each vector row's
decay/scale request, cache one unit per name, remap into requested bounds. Use
existing numerical `quantile`; aggregate validation requires scalar requests and
fixed law metadata except declared width-dependent bounds. Activation contracts
are constant, so quantile equals strict reuse there. Keep legacy validators.
[numeric] [matrixrequests] [sampler]

Label two departures from source global arrays: scalar reuse ties all row
hyperparameters, and remapping changes width-dependent values. Meta/local
scalarization preserves conditional laws, not RNG parity. A future compatible
source-global policy must retain batched shapes and reject changed contracts;
never invisibly suffix names to avoid collisions. [B] [W]

### Transaction, admission and freeze

Fork the whole context/RNG per public composite operation: both samplers, counters,
bindings and realizations. Commit only after nested sampling, aggregate admission,
construction, fitting and validation succeed. Existing sampler forks and
activation-local staging are building blocks. Rejection must leave caller RNG
and ledger unchanged; record attempted identity/reason separately, without retry
until success. [numeric] [sampler] [categories] [spec]

Bound small branch draws incrementally, then admit exact geometry before arrays.
Count outer preferences once; nested categorical records once per owner; draws,
events, vector innovations/permutations, materialized weights, fit constants,
root/innovation buffers, live ancestor states and converter copies. Enforce the
aggregate budget, not independently acceptable sampler budgets whose sum exceeds
it. Bound raw decoder lengths/shapes before allocation/replay. [numeric] [joint]
[pipeline] [calibration]

Freeze/decode must check canonical event order, exact operation coverage, absent/
duplicate/orphan references, request names/catalogs/modes, forced decisions,
destination fields and realized factors. Sharing is the same preference identity,
not matching names in separate records. Replay independent fitting against the
complete pipeline to validate constants; never derive them from emitted rows.
[numeric] [catrecords] [bindings] [source] [spec]

## One independent ancestral fit pass

Use independent reproducible calibration rows and separate root, innovation and
shift streams. Preselect shift row indices; fitted input supplies shift values.
Freeze mechanism decisions first. Matrix plain activation needs no
data wrapper fit. Traverse topologically: [M] [A] [spec]

1. Draw frozen root input or concatenate final parent states; fit/apply mechanism
   input wrapper, function and output wrapper at their actual inputs. Root input
   precedes output/pipeline; innovation is separate. [roots] [eval] [stochastic]
2. If feature importance is enabled, fit standalone Standardize over all channels,
   then apply the frozen vector normalized by its RMS. Disabled means neither.
   Reuse frozen `CalibratedStandardization` without changing it. [N] [calibration]
3. Fit L2 on that result: `1/(sqrt(mean_rows(sum_channels(x*x)))+1e-8)`,
   without width division. [A] [N]
4. Fit converter slices after L2: Kumaraswamy extrema with frozen a/b; standard
   activation input stats, shifts and output stats with frozen primitive/scale;
   identity fits nothing. [A] [C]
5. Capture readouts by value under `historical-tuple`, `historical-graph` or
   `readout-only`; add independent innovation to state, multiply state by frozen
   importance, then pass final state to children. Retain identity converters too:
   omitted converters expose final noise/importance through fallback readouts.
   [pipeline] [eval] [N]

Share application with emission, not fitting. Receipts bind actual inputs and
finalized ancestors. The old plain-selector branch skips ancestral values; the new
fitter must traverse them whenever pipelines need fitting. [fit] [eval]

Gate moments at actual normalization inputs and standard-wrapper outputs.
Finite samples or later bounded conversion do not certify an earlier stage.
Preserve valid constant/saturated draws; distinguish uncertified moments,
numerical failures and contract failures. [moments] [spec]

## Required evidence before promotion

All proposed checks are NOT_RUN. Use independent arithmetic/reference fixtures;
existing tests are behavior contracts, not new acceptance. [tests]

| Check | Discriminating oracle |
| --- | --- |
| Shared names | Force matrix/mechanism/converter fixed and power paths: identical `acts_torch`/exponent preference identities. Matrix, node and root vectors share decay/scale; nested categorical weights remain separate. Cover shared/refresh/meta/local/global. [A] [B] [W] |
| Global policy | Different widths use one scalar unit with independent log/boundary interpolation; retained label exposes scalarization. Batched reference rejects changed shape. [B] [numeric] |
| Target restriction | Exact `y` gets forced identity with no selector/a/b/activation draws; other groups remain unrestricted. Independently model both source singleton cache failures before calling them executed facts. [C] [B] |
| Atomicity/bindings | Inject rejection after selector, exponent, vector allocation, fit and validation; compare original RNG/ledger. Delete/duplicate/reorder references, change paths/names/catalogs/factors: decode rejects. [sampler] [bindings] |
| Arithmetic | Literal two-column matrix with constant column: population Standardize, outside epsilon, channel RMS normalization, whole-node L2. Constant/single-row inverse is 10000. Uniform all-zero vector is numerical failure. [A] [W] [N] |
| Ancestor counterexample | Root input `[-1,1]`, zero bias, importance 3, sigma 1, Gaussian draws `[2,-2]`, identity converter: readout `[-1,1]`, final state `[3,-3]`. Child input Standardize fits inverse `1/(3+1e-4)`, not `1/(1+1e-4)`. Arithmetic deduction; execute independently and add nonidentity variants. [N] [pipeline] |
| Root separation | Hold innovations fixed and vary root inputs, then reverse. Verify readout/state placement and stream identities through decode/matched replay. [roots] [eval] |
| Query independence | Fixed mechanism/calibration identity, varied emitted count/labels/chunks/query values: constants and overlapping rows unchanged. Hold task geometry and sampling identities explicit. [spec] [eval] |
| Compatibility | Legacy serialized recipes and fixed-seed outputs byte-identical; malformed-record rejections retained. Changed ancestor plus stale descendant constants rejected by new replay. [source] [sampler] |

## Remaining prerequisites and profile names

Next: context/ledger, ancestral fit, new dispatch/serialization/public sampling,
then local/Colab inspection. Browser/GPU/resource/predictive evidence remains
PENDING or NOT_MEASURED. [spec] [source] [authored]

Full source profiles still require RandomPoints base/function plans, reserved
latent/converter widths, categorical converters and parent multifunctions joining
the same context. Existing exogenous roots alone do not reproduce RandomFunction.
These remain required scope, not exclusions. [P] [N] [spec]

Name this a **joint fixed-function numerical pipeline adaptation**, declaring
root/mechanism coverage and independent-fit, frozen-parameter, explicit-RNG,
scalarization and target-restriction adaptations. It is neither full-prior
reproduction nor an independently sampled postprocessor. [spec] [B] [N] [laws]

[spec]: ../specs/0028-causal-network-generators.md
[laws]: 0028-node-pipeline-sampling-laws.md
[B]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py#L25-L197
[A]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L15-L217
[W]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py#L7-L56
[M]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py#L13-L97
[P]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py#L7-L70
[N]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L20-L73
[C]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py#L46-L73
[joint]: ../../src/r0/modules/priors/lib/source_matrix_sampling.py#L33-L200
[fit]: ../../src/r0/modules/priors/lib/source_causal_calibration.py#L19-L109
[sampler]: ../../src/r0/modules/priors/lib/source_activation_sampling.py#L31-L163
[planner]: ../../src/r0/modules/priors/lib/preferred_matrix_sampling.py#L36-L135
[matrix]: ../../src/r0/modules/priors/lib/source_matrix.py#L24-L96
[source]: ../../src/r0/modules/priors/lib/source_causal.py#L25-L172
[authored]: ../../src/r0/modules/priors/lib/causal_pipeline.py#L11-L40
[categories]: ../../src/r0/modules/priors/lib/categorical_sampling.py#L21-L126
[catrecords]: ../../src/r0/modules/priors/lib/categorical_preferences.py#L160-L226
[numeric]: ../../src/r0/modules/priors/lib/parameter_preferences.py#L153-L360
[bindings]: ../../src/r0/modules/priors/lib/matrix_preferences.py#L27-L83
[matrixrequests]: ../../src/r0/modules/priors/lib/matrix_preference_prior.py#L50-L82
[pipeline]: ../../src/r0/modules/priors/lib/node_pipeline.py#L18-L214
[calibration]: ../../src/r0/modules/priors/lib/calibrated_activations.py#L128-L240
[roots]: ../../src/r0/modules/priors/lib/causal_nodes.py#L162-L208
[eval]: ../../src/r0/modules/priors/lib/causal_evaluation.py#L21-L137
[stochastic]: ../../src/r0/modules/priors/lib/causal_stochastic.py#L13-L79
[moments]: ../../src/r0/modules/priors/lib/causal_moments.py#L73-L189
[tests]: ../../tests/priors/test_source_matrix_sampling.py#L11-L110
