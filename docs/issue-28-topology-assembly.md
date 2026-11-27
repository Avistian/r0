# Sampled topology and mechanism construction

This extends the standalone Cauchy sampler into the frozen graph workflow for
#28. It implements one topology family; it does not close the remaining topology,
width, readout, pipeline, noise, intervention, or workbench options in the issue.

## Public workflow

`TopologyGraphRequest` declares a `CauchyGraphPrior`, widths, observations,
targets, task controls, and the same root/function/noise/pipeline controls as
`JointGraphRequest`. `sample_topology_graph` consumes four caller-owned NumPy
streams: named, raw, calibration, and fit. Cauchy topology consumes the raw stream
first; the existing graph constructor then uses the realized parents and their
ordered channel widths. All four caller states publish only after successful
construction and validation. Failures leave all caller streams unchanged.

`SampledTopologyGraph` retains the original request, complete topology transcript,
complete sampled mechanism graph, and four initial/final snapshots. The new
`sampled-topology-graph-v1` envelope leaves the existing explicit-DAG archive
schema unchanged. `replay` reconstructs the entire construction. `with_task`
validates the original before rebuilding with changed task controls.

`recipe_from_dict` and `topology_graph_from_dict` reopen complete archives using
bounded metadata admission, caller-authorized limits, initial snapshots, and a
comparison of every retained field against replay. Missing or changed fields
are rejected, including discarded topology proposals. Input dictionaries are
not modified. Larger declared exploration caps require explicit caller limits;
this applies to inspection and pair comparison as well as decoding.

`generate_task` saves the full envelope and marks its mechanism card as
`sampled-topology-and-functions-v1`. `explain_generation` reconstructs the runtime
using that envelope and preserves requested original-row order and repetition.
`paired_recipe` and `generate_pair` retain topology and mechanism construction
while changing supported task controls. These task-level comparisons are not
hard or soft causal interventions.

Episode context size and split remain in all three nested task records in the
archive. They are excluded from source identity at those three locations, so
resplitting changes episode identity but preserves source identity and data.
Genuine task controls remain part of source identity.

## Resource contract

The outer causal limits cover the combined envelope. Before sampling, conservative
allowances for topology, request copies, and snapshots are reserved; the remainder
is passed to the existing mechanism constructor. The combined raw-draw allowance
also reserves all four topology calls, all `n*n + 2*n + 1` topology values, and
record/snapshot storage before passing the remainder to mechanisms. These are
admission estimates, not measured peak resident memory. No silent truncation,
resampling, or fallback topology is performed when a budget is exceeded.

## Verification scope

Local regressions exercise replay, archive tampering, caller rollback, combined
raw budgets, shared generation/inspection, source identity after resplitting,
paired noise controls, and reopening with explicit exploration limits. The five
supported concrete NumPy engines are covered from noninitial stream states.
Independent review and the full candidate check remain required. No GPU,
scientific distribution study, browser, Colab, or cloud run is claimed here.
