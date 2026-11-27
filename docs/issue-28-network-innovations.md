# Standalone network internal innovations and input importance

This versioned library slice composes all seven standalone architecture families
and authored combinations with internal stochastic equations. It retains the
original `NetworkRecipe`, including sampled architecture/hyperprior custody.
It does not change `standalone-network-v1`, frozen structural masks, legacy MLPs,
or the original eight generic frozen-function selectors.

## Source contracts

- [TabPFN paper v6 C.1 and C.2.3](https://arxiv.org/html/2207.01848v6):
  noise appears inside the activation; outgoing weights from each input receive
  a sampled importance weight. Per-node noise means and standard deviations are
  described, but their full distributions and the input-importance law are not
  specified. `p1-paper-placement` therefore requires explicit adapted mean laws;
  importance is authored or sampled from an explicitly selected law.
- [Pinned TabPFN implementation, lines 56–69](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py#L56-L69):
  the first affine layer is exempt; subsequent blocks apply activation, affine,
  then centered Gaussian noise. Optional frozen channel scales are
  `abs(Normal(0, base_sigma))`. `p1-778f0d0-placement` implements this ordering,
  requiring an explicit scale prior rather than assuming the paper TNLU law is
  identical to the pinned implementation's configuration.
- [Pinned TabICL MLP, lines 169–191](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_mlp_scm.py#L169-L191)
  uses that same first-affine exemption and centered activation–affine–noise
  sequence, with optional pre-sampled half-normal channel scales.
  [Pinned scalar sampler, lines 151–170](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_hp_sampling.py#L151-L170)
  has relative sigma and upper truncation, distinct from paper Table 5's
  independent mu/sigma and unbounded positive truncation. The existing
  `NetworkScalarPrior` records retain this distinction. Noise hyperlaw bounds
  are `0.0001..0.3`; source helper overrides remain visible in the request.

These profiles identify placement and explicit statistics laws, **not full
TabPFN/TabICL reproduction**. They require sequential single-expert layers and
reject unsupported gates/skips rather than disguising an extension as a source
implementation. Generic `r0-explicit` supports all architectures after each
block, plus before-activation noise on a single after-activation expert.

## Public construction and behavior

`NetworkInnovationStage` specifies channel means, nonnegative scales, placement,
`ExogenousDistribution`, and `NoiseResponse`. `None` exempts an entire block.
`NetworkInnovationPlan` supplies one stage per original block, optional input
importance, and an explicit gate-importance extension. Importance multiplies
outgoing weight columns, including direct-input projected skips. It does not
rescale observed inputs or move through nonlinear activation functions.

`with_network_innovations(source, plan)` returns a
`NetworkInnovationRecipe` with explicit functional nodes and `block_nodes`
mapping input state 0 and every original block to its final state node.
`FrozenNetworkBlockFunction` consumes only actual ordered parent/skip slices;
no duplicated ancestors or dense routing matrices are introduced. Noise and
post-noise activation stages are separate equations. Compiled node records,
readouts, task controls and source must agree exactly on decode.

`sample_network_innovations(source, request, statistics_rng=...)` samples frozen
statistics from a separate injected stream. `NetworkInnovationRequest` supports
fixed, choice, integer-uniform, uniform, log-uniform, paper TNLU and pinned
TabICL TNLU laws; shared, per-block and per-channel mean/scale scopes; optional
half-normal channel scales; and per-input importance. It records scalar
realizations, every raw draw and initial/final RNG snapshots. Unsupported source
combinations and operational caps reject before draws; failure rolls back the
caller stream. Negative sampled scales reject without retry or clipping.

Row-level innovations use the established functional engine's independent keyed
streams. Frozen masks never become per-row dropout masks. Authored extensions
include Gaussian, uniform, Student-t and mixture innovations, correlated factors,
ball geometry, nonzero channel means and signal-dependent scales. These are
explicit r0 extensions. Hard/soft state and readout interventions reuse the
existing engine, retaining unaffected streams and population identities.
Original-row inspection includes affine/activation/expert/gate/skip stages and
explicit noise equations. Full state cuts bypass the replaced functional node.

`evaluate_network_innovation_reference(recipe, observed_inputs)` evaluates the
**zero-innovation raw latent reference**, retaining mean terms. Passing
`innovations={compiled_noise_node: tensor, ...}` evaluates a **fixed-innovation
raw latent response**. Inputs already contain the observed root output. The
function neither reapplies the root nor invokes a classification adapter.
Neither reference is a conditional expectation or class probability. Supplied
innovations are standardized draws after the configured exogenous transform.
Every configured noise node must be supplied, even if its scale is zero.

## Explorer and shared local notebook delivery

The construction selector includes all seven architectures, authored sparse/gated/
skip combinations, all 28 architecture-hyperprior presets with noise/importance,
three qualified paper/pinned placement presets, and five authored non-Gaussian/
signal-dependent extension presets. JSON controls expose the complete source,
statistics request or explicit plan. A noisy construction wraps an existing
standalone source; it never silently substitutes a fixed MLP.

The architecture editor displays original blocks and a separate compiled-node
mapping. Preparation, validation, decoding, recompilation and saved generation
run in the bounded subprocess. Edits freeze an authored architecture and explicit
statistics plan; the original sampled archive and its raw records remain intact.
Invalid plans retain the editable text and accepted diagram. Frozen Next/replay
retains source, sampled statistics and raw custody while advancing exact task IDs.
Local edit export/reopen uses the existing linked archive format.

Response controls explicitly select zero innovations or fixed innovations from
the selected original row. `network_innovation_row` reconstructs every configured
noise channel (including channels beyond 32-column trace previews) from the
original keyed streams, after admission. Query geometry, finite scans, decode,
row reconstruction and evaluation run inside the guarded curve worker. Exported
JSON identifies the reference and original task/row. These raw latent responses
never claim conditional means or classification probabilities.

Compiled state/noise/activation nodes are available in the existing guarded
intervention controls and original-row before/after traces. Task-only noise and
separation pairs preserve equations and statistics, updating the source task
binding consistently. Width and importance changes remain explicit mechanism
edits; no implicit architecture rewrite is introduced.

The same HTML and `ExplorerSession` callback support local notebook delivery.
Actual Chromium local workflows are author-tested; independent UI review remains
separate. Hosted Colab execution is NOT_RUN. This implementation does not claim
complete source SCM topology/root laws, learned neural training, or integrating
nonlinear stochastic networks to obtain conditional expectations.
