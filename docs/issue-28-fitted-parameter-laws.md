# Recorded fitted-tree parameter laws

This #28 slice adds parameter sampling to the existing independently fitted DT,
ExtraTrees, RF and GBDT mechanisms. The old explicit FittedTreePlan and fitted
archives keep their schemas and stream schedules. New archives use
`sampled-fitted-functional-causal-v1` and retain each node's sampled plan alongside
its actual fitted realization. It does not replace directly sampled tree families
or complete the source-specific fitting lifecycles.

## Pinned parameter law and fidelity boundaries

The named `tabicl-pinned-depth-count-v1` profile follows the expressions in
[TabICL `_tree_scm.py` at 9530bd65](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py):

- depth `min(4, 2 + floor(E_depth))`;
- estimator count `min(4, 1 + floor(E_count))`;
- independent exponential draws with scale 2 (rate 0.5), once per selected node.

The cap retains the entire upper-tail probability. DT consumes and records the
source count draw even though its backend fits one tree per output. That unused
parameter is explicitly labelled. Generic r0 profiles reject unused parameter
laws instead of consuming meaningless options. The profile cannot override those
two laws; use the separately named explicit profile for changed rates or bounds.

**Only parameter-law fidelity is claimed.** Returned records and the explorer
identify sklearn independent-output fitting and the frozen independent-calibration
lifecycle as r0 adaptations. RF/ExtraTrees/DT have their existing explicit backend
parameters. GBDT/classifier uses of the pinned two laws are extensions, not evidence
of a documented TabICL component. XGBoost is neither installed nor aliased to GBDT.
No historical bitwise RNG or complete paper-prior reproduction is claimed.

## Broad explicit option surface

`FittedParameterLaw` supports fixed values, categorical choices with optional
probabilities, inclusive integer-uniform intervals, capped exponential integer
laws, uniform real intervals, and log-uniform real intervals. Cap arithmetic
checks the tail before integer conversion; nonfinite draws fail without retry.
Rates must be positive finite with finite inverse scale.

`FittedTreeParameterPrior(base=..., laws=((name, law), ...))` accepts these fields:

| Field | Compatible use |
| --- | --- |
| family | DT, ExtraTrees, RF, GBDT; all combined choices must be compatible |
| depth | Integer laws within the existing 1–8 envelope |
| estimator count | Integer laws 1–32 for ensembles; pinned source DT consumption is explicitly unused |
| fitting rows | Integer laws 32–1024 |
| classes | Integer laws 2–100 for classification fits |
| splitter | Fixed/categorical `best` or `random`, DT only |
| classification rule | Fixed/categorical probability mean or majority, nonboosted classification only |
| learning rate | Fixed/categorical/uniform/log-uniform in (0,1], GBDT only |

Task type and input/output widths stay explicit base-plan controls. No sampled
invalid option is silently repaired or redrawn. Worst-case numerical endpoints
and all named backend choices are admitted before draws; incompatible unused
controls reject before consuming randomness. Existing fit limits remain bounded
engineering envelopes, not measured throughput claims or a promised Cartesian
product of maxima.

## Custody, fitting and replay

`sample_fitted_tree_plan(prior, rng=..., limits=...)` returns a
`SampledFittedTreePlan`. Its one parameter stream owns initial/final snapshots and
raw values, laws, parameters, roles and state digests. Reconstructing the record
replays those draws and compares the complete resulting plan. Its source,
backend/lifecycle labels and unused-parameter declaration are validated too.

`sample_parameterized_fitted_graph(skeleton, ((node, prior), ...),
parameter_rng=..., prefix_rng=..., input_rng=..., target_rng=..., backend_rng=...)`
uses five distinct streams in one outer transaction. It samples selected plans,
then reuses the existing independent-ancestor fitter. Each plan is bound exactly
to its node's fit record and existing backend-parameter validation. Success commits
all streams; backend or final validation failure commits none. Consecutive sampled
nodes retain a checked parameter-stream chain. Before fitting, admission charges
worst-case fitting tables plus parameter raw metadata across selected nodes.

Archives retain both parameter and fitting custody and reopen without refitting.
Generation and original-row inspection use the frozen fitted functions. Mechanism
edits remain authored split-table edits; they do not claim to preserve the sampled
parameter provenance of the changed function. The original archive retains it.

## Shared explorer and remaining work

Twelve additional presets expose all four families' regression/classification
extensions and four explicitly labelled pinned-law adaptations. JSON controls
expose each base plan and law. Source details show actual sampled parameters,
profile, expression, raw count, unused choices, backend and lifecycle labels.
The common local/notebook path uses the guarded construction worker; exact decimal
IDs and original records survive Next task, replay, export and reopen.

TabICL graph overrides and actual-parent-batch refitting, O'Prior support-only
fitting, Mitra mixed categorical/Gaussian fitting populations, source-specific
noise/layer ordering, XGBoost shared-vector leaves, and broader direct/standalone
tree breadth remain required work. Independent calibration does not satisfy any
of those lifecycle requirements. Hosted Colab/GPU/scientific comparisons and
maximum-resource performance are NOT_RUN for this slice.

The subsequent [explicit population lifecycle slice](issue-28-fitted-population-lifecycles.md)
adds actual support, batch-reference and mixed synthetic populations with explicit
selected plans. The sampled-parameter API here retains its independent-calibration
contract; it is not relabelled as a paper lifecycle.

[Population-law compositions](issue-28-sampled-population-fits.md) now add a
separately versioned sampler for support, batch-reference and mixed-independent
fits, retaining effective rows and source/lifecycle distinctions.
