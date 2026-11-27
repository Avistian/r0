# Source-qualified internal network observation components

This component advances P1.graph.02–.09 in issue #28. It constructs a bounded
channel catalog and an immutable selection receipt, with strict archive replay.
The versioned observation recipe now retains this receipt through the generation
dispatcher, task archives, traces and interventions. A guarded workbench workflow
and hosted Colab verification remain required. Variable-width networks and nonsequential architectures remain explicit
future extensions; this first source profile requires equal-width sequential,
single-expert blocks without gates or skips. No failed request enlarges layers,
resamples a network, substitutes a policy, or retries a selection.

## Source contracts

[TabPFN paper C.2.2](https://arxiv.org/html/2207.01848v6#A3.SS2) describes adjacent
ordered blocks of layer nodes. `p1-paper-block-r0-v1` implements within-layer
feature windows. Uniform selection across all admissible window starts, uniformly
permuted distinct internal targets outside that window, and exclusion of a forced
final target suffix are explicit r0 reconstruction choices, not claimed paper
hyperparameter laws. Its unrestricted selection and optional feature permutation
are separately requested alternatives to the adjacent ordered-block policy.

[Pinned TabPFN prior 778f0d0, lines 118–137](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py#L118-L137)
is a different policy. `p1-778f0d0-v1` excludes root causes and the first block's
output, corresponding to the source's first affine output, then concatenates the
remaining propagated block channels. Block selection draws a contiguous window
of feature-count plus output-count channels and permutes that window. Unrestricted
selection permutes all catalog indices except the final index. Both reserve the
first output-count slots for targets, even when final targets override those slots.
`preserved` sorts feature indices; `permuted` retains sampled order. This can select
a forced final target as a feature in block mode. The receipt's
`target_feature_overlap` reports that source behavior without silently filtering it.
The profile qualifies the observation policy, not all source generator settings:
a supplied block can have an activation, and its actual propagated state is used.

Selection uses injected NumPy generators and retained raw integer/permutation
draws. It reproduces these policy operations, not Python/Torch RNG bitstreams.
There is no inferred random law over feature count, output count, target mode,
feature order, or network widths. The caller supplies them explicitly.

## Public interface and custody

`NetworkObservationRequest` retains the full `NetworkRecipe` or
`NetworkInnovationRecipe`, selection controls, and raw limits.
`network_observation_catalog` maps each eligible `(block, channel)` to the actual
compiled propagated node, including post-block noise and activation placement.
It uses the same compiler as stochastic network generation; it does not relabel
an affine intermediate as a propagated output.

`sample_network_observations(request, rng=...)` returns
`SampledNetworkObservations`: the complete source request, catalog, ordered
feature/target indices, initial/final RNG snapshots, and all raw draws. Request
admission precedes RNG transaction/snapshot construction. Raw draw counts,
retained values, metadata, catalog size, source graph workspace and snapshot
reserves are bounded. Failed final receipt validation leaves the caller RNG
unchanged. Reopening requires exact fields, primitive types, catalog references,
raw laws/values and final state. Caller limits apply before replay allocations.

`network_observation_request_from_dict` and `network_observations_from_dict`
accept ordinary JSON-compatible objects or `dataclasses.asdict` output. Component
versions are `network-observation-request-v1` and
`sampled-network-observations-v1`. Existing recipe versions are unchanged.

`receipt.runtime(task)` explicitly materializes a `FunctionalCausalRecipe` with
selected channels as features and internal/final channels as targets. The supplied
task must have matching feature count and regression/classification output width.
The runtime supports existing generation, traces and recipe persistence, but its
archive alone contains the frozen graph, **not this construction receipt**. Retain
and reopen the component receipt separately for source selection provenance.
Use the versioned task envelope below to retain source selection custody.
Workbench controls and shared local/Colab workflows remain required follow-ups.

## Pinned overlap materialization repair

A pinned final-target draw can select the same propagated channel as an observed
feature. Materialization now adds one explicit identity target node for each such
channel, retaining the selected feature and original receipt unchanged. The alias
copies that channel exactly and is a descendant of its original node; intervention
on the original state therefore flows to the alias. This preserves the source's
possible target leakage rather than filtering or resampling it. Generic graphs
still reject identical observed/target references. Alias matrices, channels and
working storage are admitted before construction. The task envelope exposes original references, overlap and derived aliases in
its mechanism card and original-row traces.


## Task envelope and intervention delivery

`NetworkObservationRecipe(task, observations)` stores the complete immutable
selection receipt and task controls. Its `network-observation-recipe-v1` JSON
contains no independently editable duplicate graph: nodes, observed references,
original targets and identity aliases are derived from the retained source and
selection. `recipe_from_dict`, `generate_task`, task ZIP save/reopen and
`explain_generation` preserve this record, exact seed/task IDs, raw construction
receipts and original row order (including repeated selected rows).

Admission counts full nested metadata and reserves retained/raw construction
storage, sampled network/hyperprior/statistics/root/prototype replay and compiled
graph emission before restoring any construction RNG. Caller limits cannot be
expanded by source-declared limits. Source geometry is previewed without replay;
strict component decoders then verify every retained draw and derived field.
The envelope does not resample selections or silently shrink a source's limits.

Causal comparisons accept the envelope as their original base. Hard state,
readout and both-location interventions and soft equation/root replacements use
the same compiled nodes and preserve the complete original receipt. For a target
alias, a readout-only change to its parent changes the captured feature but not
the downstream target; a state change propagates to the alias. Cards/traces keep
original and effective target references and explicitly report source target
leakage. The materialized runtime remains a generic functional graph; this does
not add native categorical observation semantics.

Exact replay applies at the same row shape for every admitted source. Existing
coordinate-root v1 laws consume whole coordinate arrays sequentially, so changing
the row count can shift later-coordinate RNG streams. The envelope preserves
these versioned semantics, frozen normalization and exact same-shape replay; it
does not claim universal prefix stability. A versioned prefix-stable extension
remains required where the broader specification calls for it.

Broad follow-up scope is unchanged: variable-width and nonsequential/gated/skip
observation policies, source-qualified additional selection laws, interactive
guarded workbench editing/inspection/export, and shared local/Colab workflows.
The existing sequential profiles compose with sampled weights/hyperpriors,
internal innovations/importance, coordinate roots and prototype roots; they do
not silently substitute another architecture when a source is unsupported.


Saved envelope and envelope-based intervention traces validate complete task
identity, task type/classes/schema/origin, tensor shape/dtype/layout and values,
probability presence, splits, full mechanism metadata, recipe identity and
execution schema before slicing original rows. Implementation identity is bound
to the recipe. Historical library/platform strings, positive thread counts and
recognized CPU/CUDA provenance may differ from the current CPU replay environment;
exact saved tensor replay is still mandatory.
