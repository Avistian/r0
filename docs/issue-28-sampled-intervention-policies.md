# Sampled intervention policies and retained environments

This library/graph/task slice extends the existing authored intervention operators.
It does not complete #28, add a workbench, or reproduce TAFFY's full task mixture.
Inventory `TA.intervention.01–09`, `SPEC.I01.01/.02`, and `R0.extension.21/.22`
remain distinguished at operation and sampling-law level.

## Source and reconstruction

[TAFFY v1 Appendix A.3, equations 14–16](https://arxiv.org/html/2610.07559v1#A1.SS3)
selects non-label nodes uniformly without replacement, at count
`ceil(.05*(m-1))`, including hidden nodes. Hard values are uniform within the base
table's node extrema, constant across an environment. Soft changes resample
parameters within the same mechanism family, retaining parents and exogenous
laws. The source regenerates a fresh exogenous population. It does not specify
every mechanism's parameter distribution.

`taffy-selection-r0-laws-v1` implements that eligibility/count law for scalar
functional graphs with one explicit target node, no numerical pipelines, and
fresh population comparison. The label exclusion must equal the graph's target-node
set. The profile name does **not** claim a published distribution for resampled
function parameters or independent calibration.

`explicit-policy-r0-v1` additionally permits ordered eligible subsets, fixed count
or fraction, multiple excluded target nodes, vector nodes, numerical pipelines,
explicit state/readout/both locations, and matched exogenous comparisons. Invalid
counts or insufficient eligibility reject; no replacement selection or repair.

## Public interface

- `InterventionPolicyRequest(base, label_nodes, ...)` declares source/profile,
  hard or soft operation, eligibility, count/fraction, location, population identity,
  reference lifecycle and resource caps.
- `InterventionParameterLaw(node, path, family, ...)` declares one numeric group in
  a frozen equation. Paths use field names and decimal sequence indices. Examples:
  `("matrix",)`, `("leaves",)`, or
  `("blocks", "0", "experts", "0", "matrix")`. Groups must contain finite
  floating parameters; integer indices, flags, class maps and versions cannot be
  resampled. Paths cannot overlap. Every eligible soft node requires a declared law.
- `sample_intervention_policy(..., selection_rng, value_rng, calibration_rng)`
  uses three distinct caller-owned streams atomically. It retains the initial/final
  snapshots, selection permutation, parameter draws and separate keyed reference
  raw ledger in `SampledInterventionPolicy`.
- The receipt works through `generate_task`, `recipe_from_dict`,
  `explain_generation`, `generate_policy_pair` and existing causal pair save/load.
  Extracting `.runtime` intentionally gives only the authored intervention; keep
  the envelope when sampling provenance is required.

Soft laws are explicit r0 conditional parameter resampling: normal
`location + scale*Z`, uniform `[location, location+scale)`, log-uniform
`[location, scale]`, and uniform selection from `choices`. Choice values are
reconstructed as floats. All selected groups redraw independently. Frozen
architecture, activation selection and other unlisted parameters remain fixed.
Soft resampling acts before the existing pipeline/readout; readout-only soft location
aliases reject. Hard-bound controls and the calibration stream are unused for soft
policies; their card reports `not-used-soft` and no reference seed.
The rebuilt frozen equation must preserve its type and input/output widths and
pass normal schema/geometry validation. Invalid sampled domains reject atomically,
without retries, clipping, normalization or family substitutions. Four different
function families and all four laws have public coverage; the generic path seam
also accepts other registered frozen function records when their declared groups
satisfy the same contract. This is **not** backend refitting or an inferred prior
from one matrix. Hyperparameter/family-prior resampling remains required separately.

## Reference bounds and semantics

`source-reference-population` uses all `base.task.rows` at explicit reference
seed/task identity and the original root/exogenous key protocol. This is the
retained source base-table convention. `independent-calibration-r0` instead draws
an independent seed and uses a distinct `policy-independent-reference-v1` namespace
for an explicit fixed row count. Query rows never enter either bounds fit. `with_task` keeps independent bounds
unchanged when query row budgets change; the source-reference variant explicitly
reconstructs its whole base population and bounds for the new row count.
The complete reference raw ledger and chosen-node bounds are retained and replayed.
Constant ranges remain valid. One hard quantile per chosen channel is drawn once
per environment, even for constant ranges.

For readout-only changes, bounds describe readouts. State and state/readout changes
use propagated-state bounds; the extension explicitly clamps both to the same
sampled value when requested. The scalar source profile has no earlier captured
readout ambiguity. Unselected equations, parent sets, root laws and exogenous laws
remain those of the base. All parameter draws are environment-level, not per row.

`population` means fresh keyed exogenous draws; equal row positions are not the
same units. `matched` retains keyed variates for the existing causal execution
contract. Pair cards retain that distinction and the policy receipt. Original
base archives and policy archives remain separately inspectable. Trace control
nodes expose selected targets and sampled parameters in addition to original-row
intervention traces. Inspection verifies semantic metadata, array dtypes/values,
complete task archive and execution implementation before adding trace nodes;
descriptive historical runtime fields remain provenance, not reverified hardware.

## Admission, replay and boundaries

Admission includes the complete frozen request, original/reference graph geometry,
all eligible parameter groups, conservative complete raw reservations and snapshot
storage **before** caller stream transactions. Bounded innovation rejection laws
reserve their existing finite maximum draw budget. Root-record parameter counts
cover retained prototype and source-root complexity; no bank is silently reduced.
The conservative bound may refuse a request even if its selected branch would fit;
limits are explicit controls, not scientifically supported maxima. Decoder metadata
and actual retained raw costs are checked before restoring policy snapshots, then
complete deterministic replay enforces exact primitive types and every field.

The current request accepts authored `FunctionalCausalRecipe` records containing
registered frozen equations. Sampled/fitted graph envelopes, native categorical
and converter-domain policy samplers, architecture/hyperprior resampling, adaptive
bounds policies, environment mixture assembly, workbench and shared notebook controls
remain required follow-ons. Existing manual typed interventions remain available;
this slice does not relabel them as sampled policies. Source assembly/postprocessing
belongs to its existing sibling-ticket boundary. Hosted/GPU/scale/diversity evidence
is not established by these library checks.

Nested graph limits remain binding during policy admission: the reference graph
and complete before/after comparison use the componentwise minimum of the
original base limits and policy limits. Admission probes every eligible change
before sampling a subset, using geometry-preserving constant clamps or original
function parameters. This deliberately conservative check can refuse a request
whose smaller sampled subset would fit; it never widens the base budget, silently
reduces the count, or rewrites retained base records. Policy raw/receipt storage
also remains bounded by the separate aggregate policy limits.
