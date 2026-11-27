# Sampled structural innovation pipelines

`JointInnovationPipelinePrior` is an opt-in `r0-innovation-pipeline-prior-v2`
record. It preserves the numerical converter pipeline and samples the complete
structural innovation law. `JointPipelinePrior` v1 and its Gaussian sigma switch
retain their fields, serialization and draws. V2 rejects that legacy switch to
prevent applying two noise scales.

The v2 `innovation` field accepts `JointInnovationPrior`. Its named decisions use
the same construction context as the graph, functions and converters: local,
global and meta modes, categorical sharing/refresh and numerical global policies.
Catalog values form part of each named request's identity; different catalogs do
not silently reuse the same categorical contract.

| Family | Draw before a factor |
| --- | --- |
| `gaussian` | Gaussian, then selected location and input scale |
| `uniform` | Uniform[-sqrt(3), sqrt(3)], then location and input scale |
| `student_t` | Student-t with selected df, without variance rescaling, then location/scale |
| `mixture` | Gaussian plus an independent equiprobable signed gap per channel, then location/scale |
| `ball` | Standard Gaussian direction normalized to unit length, radius U^(1/width) |

Ball is a complete geometric law; it does not draw and discard an independently
selected base family, location, scale or truncation. Other families support optional
conditional truncation of the standardized draw, before location/scale. This reuses
the existing explicit128-batch rejection budget and minimum256-value batch. It is
not task rejection, post-hoc clipping or variance standardization.

The factor catalog contains identity and Gaussian. Identity stores no matrix;
Gaussian draws F entries from N(0,1/width), retains every raw entry, and applies
row draws @ F. F need not be orthogonal or full-rank. For a standard Gaussian base,
covariance is FᵀF; it is not asserted to equal F. Ball covariance before transformation
is I/(width+2). Factors remain separate from structural scale.

Default configurable extension catalogs/bounds are:

- Five families; identity/Gaussian factors; truncation None,2,4,8.
- Student-t degrees0.75,1,1.5,2,3,5,10; mixture gap[0.25,4] on log scale.
- Location[-0.5,0.5] on linear scale; input scale[0.25,2] on log scale.
- Structural scale[1e-5,1] on log scale; response offset fixed1;
  strengths0,0.1,0.5,1; powers0.5,1,2.

Bounds may collapse to a fixed value. These are r0 profile choices, not paper
probabilities. Local mode samples uniformly in the selected linear/log parameter
space; global/meta behavior follows the retained shared preference contract.
`scale_scope="node"` draws one structural scalar and repeats it over channels.
`"channel"` records separate scalar draws under the same named preference. Global
scalar preferences may therefore produce equal channel values; independent local
channel draws need not. An offset0/strength0 response explicitly disables innovation.

## Exact stage and fit semantics

Let f be the frozen function output, P its pipeline-converted state, Z the full
exogenous draw and I the frozen node importance. The equation is:

```
scale = structural_scale * (offset + strength * abs(f)**power)
state = I * (P(f) + scale * Z)
```

The response uses f **before** pipeline standardization/importance/conversion.
Converters capture their declared readouts before this node's innovation and
importance. The three numerical converter variants retain their previous state
versus readout semantics. Descendant functions and their independent fitting
populations use the final state, including ancestor innovation and importance.
The current node's own pipeline fit sees its bare function/root output. Neither
query rows nor target labels fit the law, factor, pipeline or response.

Existing moment certificates apply to the realized input law and response at each
normalizing stage. Unsupported heavy-tailed normalization fails explicitly and
atomically; finite fitting rows do not certify population moments. Turn off
incompatible normalization to retain valid unnormalized heavy-tailed mechanisms.
No failed graph is silently retried or replaced.

Four caller streams are staged. Plans, complete frozen noise records, raw factors,
ancestral fit draws and before/after states are retained. Geometry admission reserves
factor copies and prefix/rejection-buffer floors before materialization; the shared
raw ledger additionally checks every draw and aggregate limit. Full archive decode
replays construction and binds the complete noise law, not only its scalar scale.
Admission is a conservative allocation estimate, not measured peak RSS.

## Workflows and scope

The explorer offers five labeled r0 presets and controls for family/factor catalogs,
structural scale scope/range, Student-t degrees, conditional truncation and response
strength/power. Full JSON also exposes location/input-scale/mixture-gap/offset bounds.
Presets disable optional normalizing stages and converter warps so heavy-tailed
families can be explored explicitly. Users may enable compatible stages in JSON.
Construction provenance names the r0 extension and exact response timing. Actual
traces expose exogenous draws, scales, captured readouts and final states.
Save/reopen/edit/download and Colab callbacks use the shared public decoder.
Mechanism edits remain distinct from causal interventions. Hosted Colab is NOT_RUN.

`I2.node.05` remains the separate source Gaussian post-capture law. This expansion
addresses accepted spec13, SPEC.R02 and R0.extension.20 without claiming unspecified
paper innovation laws. Source-specific P1 noise hyperpriors, other root families,
all I2 categorical converter profiles, and their typed/intervention integration
remain required follow-on work in [the implementation plan](issue-28-pipeline-family-plan.md).
This checkpoint does not claim complete #28 acceptance or source reproduction.
