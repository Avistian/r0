# Explicit causal interventions and replayable comparisons

The public `HardIntervention`, `SoftIntervention`, `CausalInterventionRecipe` and
`generate_causal_pair` APIs operate on frozen functional or native DAGs, including
their sampled construction archives. A new versioned intervention recipe retains
the complete original base graph. It does not turn a mechanism edit or a task
noise/separation pair into evidence of a causal intervention.

## Locations and equations

A hard continuous intervention names a node, ordered channel indices and one finite
constant per channel. A scalar is one channel; a vector lists all selected channels.
Several disjoint interventions may act simultaneously. Duplicate locations,
hard/soft overlap at a node, invalid widths and invalid native labels reject.
The location is retained in recipes, comparison cards and original-row inspection:

| Location | Propagated descendant state | Observed/target readout |
| --- | --- | --- |
| `state` (default) | Selected final state channels are clamped. | Channels without a converter capture follow that same state; earlier captured converter readouts stay unchanged. |
| `readout` | Unchanged. | Selected measurements are clamped. This is explicitly a readout intervention, not a causal-state do-operation. |
| `state_and_readout` | Selected final state channels are clamped. | The selected captured measurements are also explicitly replaced. |

Consequently `do(state)` is not presented as `do(target)` when the target points to
an earlier converter capture. A readout-only clamp cannot affect descendants. An
explicit readout clamp takes precedence over state-to-readout aliasing, regardless
of the order of the intervention records. Root and internal-node interventions use
the same location rules.

When every required state and readout channel is replaced, the original function,
root/innovation draws and pipeline are bypassed. An overflowing overwritten
function therefore need not invalidate standalone surgical generation. A matched
pair still needs a valid observational baseline. Partial-channel coupled functions
and pipelines are evaluated in full, then selected locations are overwritten;
failures in that evaluated coupled computation are not silently discarded. A
state-only intervention with an earlier captured readout still evaluates the
original computation required by that retained measurement. Trace metadata names
bypassed nodes and the evaluated partial-channel contract.

For a native discrete node, `HardIntervention(node, category="declared label")`
replaces the emitted category and the complete one-hot state consumed by children.
It bypasses the original equation and emission, and retains the original vocabulary.
Forced target probabilities and scores are one-hot indicators, explicitly labeled
as such rather than reported as original logits. Native continuous variables expose
one emitted state; intermediate pipeline captures are not separate native variables.

A soft intervention replaces an optional frozen function, noise-scale vector,
signal-dependent noise response, or exogenous law at one node. Input/output geometry,
parent order, domain, pipeline and independent calibration remain fixed. This includes
coefficient changes and nonlinear/fitted-function replacements represented by their
public immutable function records. The original backend fitting provenance stays
with the original base archive; a replacement is not labeled as a refit.

## Matched and population comparisons

`generate_causal_pair(base, interventions=(...), mode="matched", seed=17, task_id=3)`
retains the original row positions, split, unaffected mechanisms and keyed root,
innovation, native-emission and target-adapter streams. Comparison replay verifies
unaffected root values, structural innovations and emission uniforms where both
sides evaluate them, plus shared stream-initial fingerprints. Bypassed nodes have no
after-side draw and are listed explicitly. An explicitly replaced exogenous law is
excluded from an equal-innovation assertion; other unchanged draws remain coupled.

`mode="population", population=9` uses a separate declared key for newly drawn
exogenous populations, without resampling mechanisms or fitting normalization.
Row positions and splits remain comparable, but they do not identify the same units.
Cards report the actual draw equalities and observed input/target mean changes;
those finite-sample summaries are not causal estimates inferred from observations.

Frozen activation/pipeline constants never fit emitted queries. Optional numerical
target calibration is reconstructed from the original graph's separately keyed
population and used on both sides. It is not recomputed from the intervened graph,
query values, query count or context/query partition.

## Replay, inspection and persistence

Both standalone `generate_task(CausalInterventionRecipe(...))` and pair generation
use the shared numerical/native evaluators. `recipe_from_dict` decodes the complete
versioned intervention recipe under caller limits. `explain_generation` reconstructs
selected original rows, including repeated/out-of-order indices, and exposes function,
innovation, state, captured readout, forced probability and explicit changes as
applicable. The original equation and intervention are both retained.

`generation.save_causal_pair(directory, pair, max_bytes=...)` writes separate
write-once before/after ZIP groups using the original task identity. `max_bytes`
bounds each group. A final `pair.json` binds both group receipts and the comparison
card. `load_causal_pair` verifies receipts and complete pair replay; missing manifests,
changed arrays/cards and overwrites reject. Interrupted partial directories remain
without a completion manifest. Predictor archives retain the established split:
query labels, noise and diagnostic truth are not exposed as predictor inputs.
Caller-owned larger causal limits can be supplied to comparison, persistence,
decoding and inspection.

Admission counts retained base/change records, both graph work estimates and
comparison/trace buffers before generating arrays. These bounds are conservative
estimates, not measured peak RSS. There is no retry, clipping, backend fitting,
installation, network or cloud execution in these APIs.

## Worked oracle and remaining scope

For `A=U_A`, `B=2A+U_B`, `C=-A+3B+U_C`, the retained public probe checks
`do(B=4)` against `C'=-A+12+U_C` and a soft coefficient change against
`B'=-A+U_B`, `C'=-A+3B'+U_C`. Tests also distinguish propagated state from captured
readout, partial vector clamps, simultaneous changes, native categories, population
draws, overwritten overflow, fixed target calibration, malformed archives and
persisted pair replay. Raw logs and the base-RED/current-GREEN program are ignored
under `docs/evidence/issue-28-causal-interventions/`.

This implementation supports explicit CPU execution. GPU agreement, workbench/Colab
controls and saved UI comparisons are still pending. Legacy non-functional
`CausalRecipe` conversion, direct root-law/domain-emission/pipeline-component soft
replacement, stochastic/per-row hard assignments, surgery inside coupled functions,
and additional intervention locations remain extensions to implement rather than
claims made by this API. The broader accepted #28 inventory remains required; this
bounded core slice does not close the issue or replace independent verification.
