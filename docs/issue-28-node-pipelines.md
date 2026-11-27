# Issue 28: causal node pipelines and numerical state/readout conversion

This increment makes ordered authored pipelines available through shared causal
generation, decoding, original-row inspection and saved explorer sessions. It is
the integration surface for the next sampled-pipeline increment. Independent
ancestral fitting, named shared shape choices, source pipeline profiles and their
unfiltered census are still required; an authored pipeline does not claim those.
The full accepted #28 scope remains in force.
The subsequent [root-input and activation-converter increment](issue-28-root-input-converters.md)
extends the authored surface; the checkpoint evidence below remains historical.

## Source order and explicit variants

The pinned [TabICL node function](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L44-L73)
places optional standardization/channel importance and L2 before converters, then
adds Gaussian innovation and whole-node importance to the propagated state.
Converters expose separate columns. The [source audit](research/0028-wrapper-source-laws.md)
records the historical view-alias discrepancy. Implementation is independently
written; no upstream implementation is copied.

`CausalPipelineRecipe` is an authored `causal-v2` recipe with one immutable
`CausalNodePipeline` per node. Activated, stochastic and matrix nodes are accepted,
with their noise position explicitly `after_output`. Each node evaluates:

```text
frozen input activation → affine projection → frozen output activation
  → ordered frozen stages → per-channel importance → optional frozen L2
  → numerical converter state/readout branches
  → add structural innovation to state → multiply state by node importance
```

The normal stochastic noise response still depends on the pre-noise affine
response. Its broader exogenous laws and signal-dependent scaling are explicit
r0 extensions to this source pipeline order. The authored stage list also permits
broader compositions; it does not claim each composition is a published recipe.
At this checkpoint an authored root had its deterministic bias before the declared
after-output innovation. This is not the source's separate RandomPoints root draw
followed by later Gaussian innovation. Sampled source integration must retain
both random inputs at their distinct positions; the current single root
innovation is not claimed to reproduce both.
Signed and zero channel-importance factors are allowed; whole-node importance is
finite and positive. Channel factors are recorded realizations, not silently
renormalized or sampled. The source standardization/importance sampling law remains
part of the upcoming sampled construction.

At this checkpoint each `NumericalConverter` selected one unique channel and an identity or frozen
one-channel Kumaraswamy transform. Let x be that channel before conversion and K
its frozen warp:

| Variant | State passed to later innovation/importance | Captured readout |
| --- | --- | --- |
| `historical-tuple` | K(x) | copied x |
| `historical-graph` | K(x) | copied K(x) |
| `readout-only` | x | copied K(x) |

The first is an explicit reconstruction of the historical tuple intent. The
second implements the pinned graph's alias consequence by value. The third is
the corrected readout-only surface for #30, without implementing #30 observation
policy. Stored readouts never depend on accidental tensor aliasing.

Only explicitly converted channels are captured before later innovation and node
importance. Unconverted observed/target channels use their final structural state.
Descendants always consume propagated state. Thus a converted feature can remain
constant while its node's noisy state affects a descendant. No context/query
conditioning operation is introduced. The same channel cannot be simultaneously
an observed feature and target under the inherited recipe contract.

## Calibration, moments and provenance

Fitted stage, L2 and warp records retain their existing calibration constants,
identity and digest. Generation, original-row inspection and JSON loading do not
refit them. Authored records still cannot prove that a caller-owned calibration
sample was independent, or reconstruct arbitrary input samples from their digest.
Sampled construction must supply the independent ancestral stream and validate it.

Pipeline normalization checks the conservative finite-moment certificate at the
actual input of each L2 or fitted standard-activation stage. A fitted standard
activation also requires certified finite output second moments. Fixed finite
sample values cannot justify normalizing an uncertified heavy-tailed population.
A bounded warp before L2 can establish the needed certificate; later structural
noise cannot change the already captured readout's moment order. Unknown moments
remain uncertified. The check is sufficient, not an exact symbolic integrability
solver; it does not estimate moments from emitted rows.

Target normalization, when requested, fits on its existing independent calibration
stream using the target **readout**, not an interchangeable descendant-state value.
The same readout-aware moment certificate gates it. New pipeline ancestry tracks
potential exogenous dependence separately for state and readout, excluding later
innovation from captured columns. These are structural diagnostics, not an
empirical conditional-prediction or causal-identification claim.

## Inspection, admission and saved delivery

Original-row inspection exposes each ordered stage, channel importance, L2,
converted state, captured readout, final node state and structural innovation.
Observed-input and target edges point to readouts; descendant edges point to state.
Equations, frozen records and row indices accompany the realized values. Existing
preview column limits remain explicit; full records stay in the saved recipe.

Pipeline metadata counts channel width and node importance, each converter's
channel and transform constants, all stage constants, channel factors and L2
constants. The working estimate adds buffers for captured readouts, state,
inspection stages and active transform buffers, summing the reserves across all
nodes because a trace can retain every node's stage values. These conservative array
estimates are not measured process RSS or hard OS enforcement. Admission applies
before evaluation tensors. Caller-owned fitted samples retain their separate
fitter admission requirements.

Legacy recipes serialize no pipeline fields and retain their existing arithmetic.
The common evaluator was separated into its own module to keep files within the
repository's size limit; its prior internal import remains available to existing
callers. Independent legacy verification is required before acceptance.

The public decoder, generate/inspect API and ExplorerSession use the same pipeline
records. Actual browser and Colab interaction evidence remains separate and
PENDING. These records provide no claim of GPU, supported-size or scientific
acceptance.

## Worked example and verification state

With root x=0.5, K(x)=x², no innovation, root importance 2 and child weight 4:
`historical-tuple` emits feature 0.5 and child target 2; `historical-graph` emits
0.25 and 2; `readout-only` emits 0.25 and 4. This distinguishes both readout and
descendant behavior. The underlying range warp retains its declared 1e-30 epsilon;
the example agrees to the retained floating-point tolerance.

Development checks include this example, ordered-stage arithmetic, readout-aware
heavy-tail moments and ancestry, actual-input L2 admission, exact/minus-one resource
limits, malformed records, JSON loading and nine saved workflows across the three
converter variants and task types.
[Raw development logs](research/0028-node-pipeline-development.json) retain the
missing-API, readout-moment/ancestry, stage/trace and actual-input-normalization
REDs and their subsequent GREEN runs. A later budget correction sums each node's
pipeline reserve, because tracing can retain all node stages simultaneously.
The worked two-node fixture accounts for 17 parameters and 9,344 estimated bytes;
exact limits pass and one-less budgets reject. These are working-tree logs, not
immutable snapshots of every intermediate source. The manifest preserves that
qualification and distinguishes the earlier saved-workflow run from the final
accounting regression. Reproduce the current targeted checks with:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/existing/venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_pipeline.py \
  tests/explorer/test_pipeline_session.py
```

The [standards review](research/0028-node-pipeline-standards.txt) finds no hard
violations and two advisory duplication risks: evaluator finalization and
converter-variant decisions shared between runtime and moment calculations.
The scoped source review is separate from later publication additions.

## Full checks and independent acceptance

Full `make check` at `ae9b3b8b64c226d1532e7cb7939ab99b907fc2ad` passes
**770 tests, one skipped**, in 400.40 seconds, with all static gates passing.
[Full output](research/0028-node-pipeline-check-ae9b3b8.txt).
The skip is the existing CUDA case on this CPU environment. Smoke/eval-smoke are
NOT_AVAILABLE: the [current CLI help](research/0028-node-pipeline-cli-help.txt)
has no train/eval commands. Publication preserves the checked source/test bytes.

The [independent spec review](research/0028-node-pipeline-spec-review.txt) gives
scoped ACCEPT with no implementation findings. Its 81 recorded checks include
**50 independent numerical comparisons**, 20 expected malformed-record or
compatibility rejections, nine saved workflows, moment checks and ancestry checks.
NumPy equations reconstruct mixed captured/unconverted channels, all three
variants, signed/zero factors, L2, Gaussian roots, matrix descendants and
affine-dependent innovation. Maximum absolute error is **8.88e-16**, within the
predeclared `rtol=atol=2e-12`.

The supplemental [audit](research/0028-node-pipeline-audit.json) independently
counts **87 parameters and 32,768 estimated bytes** for its mixed fixture.
Pipeline reserves contribute 46 + 43 channels, plus 24 stochastic scratch channels.
Exact budgets admit and minus-one budgets reject. It also checks readout/state
mutation independence and unchanged frozen records across context/calibration-row
controls. These are declared-array estimates, not measured RSS or scale acceptance.

All **168 legacy proposals** across 28 profiles, three graph families and two task
ids retain identical recipes, tasks, diagnostics, traces and statuses against
`f5efdc3`: 164 successes and four unchanged explicit failures. Nothing was dropped
or retried. Imported-source custody binds 76 baseline modules, 80 legacy candidate
modules and 100 numerical/workflow modules to their pinned Git bytes.

[Raw artifacts](research/0028-node-pipeline-spec-artifacts.json) retain 192 logical
files in nine ordered ZIP byte parts, including exact programs, inputs, outcomes,
saved sessions and the complete original manifest. One byte-identical legacy
capture uses `duplicate-paths.json` to reconstruct its second path. Verify part
and joined-archive hashes, extract, restore the duplicate and reconstruct the
153 excluded baseline source files with Git before checking the full manifest.
The local originals remain unchanged. Four final programs in
[`scripts/verify_node_pipelines/`](../scripts/verify_node_pipelines/) are
byte-identical to executed copies. [Reproduction instructions](research/0028-node-pipeline-reproduction.txt)
require copying them into a fresh evidence directory because outputs are relative
to the program location. The initial successful audit and its later portable
revision are both retained; the final portable program was actually executed.

## Source coverage and next integration

The authored identity/Kumaraswamy converter surface advances `I2.converter.01`,
`I2.converter.02`, the newly explicit `I2.converter.02.graph`, and the
`I2.measurement.01` interface. It does not close their full sampled/environment
evidence. Activation converters `I2.converter.03` / `.03.graph` and the corrected
activation interface remain required; an arbitrary upstream stage is not their
substitute. The inventory now distinguishes both historical graph-alias variants,
with 583 source/extension/disposition records plus 140 specification anchors.

The [sampling-law follow-up](research/0028-node-pipeline-sampling-laws.md) establishes
separate root inputs and later innovations, source per-pass parameter redraws,
shared matrix/channel-weight preferences and target-restricted selector catalogs.
Freezing those parameters for independent calibration is an explicit r0 adaptation.
Sampled bindings, root-law integration, standalone source standardization, source
profiles/census, wider #28 mechanisms, typed states, interventions and actual
browser/Colab/GPU/scale delivery remain required. This is component acceptance,
not full issue acceptance.
