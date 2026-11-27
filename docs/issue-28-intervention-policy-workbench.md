# Sampled policy workbench

This slice exposes the library policy contract in
[Sampled intervention policies](issue-28-sampled-intervention-policies.md).
It adds a separate sampled-policy panel beside authored hard/soft interventions.
The existing mechanism editor remains distinct.

On an authored numerical functional dataset, prepare a hard or soft policy draft,
choose source selection or explicit r0 eligibility/count, and choose hard-bound
reference lifecycle. The editable JSON retains all policy knobs and three exact
stream seeds. Soft drafts require explicit parameter paths/laws for every
eligible node; normal, uniform, log-uniform and discrete choice laws are supported.
The guarded worker strictly admits the original archive, realizes the policy,
verifies replay and saves both datasets and the complete sampled receipt.
Invalid drafts, timeouts and resource failures preserve the active dataset and
previous valid comparison. Attempt receipts are retained.

The comparison exposes actual trace dependencies, selected nodes, bounds,
parameter laws, raw draw counts and matched/fresh semantics. Download/reopen use
the shared causal-pair persistence contract. Open a sampled result as the active
dataset to use ordinary Next and replay: the policy environment remains fixed
while Next advances the query task identity. Independently calibrated references
do not fit the query rows. Reconstruct a policy explicitly to resample its choices.
The saved raw recipe remains the authority for exact primitive values and RNG
state; rendered metadata never replaces that archive.

The same page and session run through the local notebook callback; hosted Colab
execution is NOT_RUN. Native/converted graphs, sampled/fitted graph construction
envelopes and architecture/backend hyperparameter policy sampling remain required
follow-ons. No claim of complete TAFFY task-mixture reproduction or full #28
acceptance follows from this workbench.
