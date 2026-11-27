# Issue 28: causal node pipelines and numerical state/readout conversion

This increment makes ordered authored pipelines available through shared causal
generation, decoding, original-row inspection and saved explorer sessions. It is
the integration surface for the next sampled-pipeline increment. Independent
ancestral fitting, named shared shape choices, source pipeline profiles and their
unfiltered census are still required; an authored pipeline does not claim those.
The full accepted #28 scope remains in force.

## Source order and explicit variants

The pinned [TabICL node function](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L37-L65)
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
Signed and zero channel-importance factors are allowed; whole-node importance is
finite and positive. Channel factors are recorded realizations, not silently
renormalized or sampled. The source standardization/importance sampling law remains
part of the upcoming sampled construction.

Each `NumericalConverter` selects one unique channel and an identity or frozen
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
converter variants and task types. Full checks and independent review are PENDING.
Raw development failures and current commands will be retained with publication;
this document is not yet an acceptance report.
