# Issue 28: recent option delivery and remaining breadth

This supplements the [atomic delivery map](issue-28-atomic-delivery.md), not the
[complete inventory](research/0028-source-option-inventory.md). All inventory IDs
and requirements remain in scope. These are component-level mappings, not claims
that a paper prior, its hyperparameter distribution or its trained checkpoint has
been reproduced. Unlisted options remain NOT_AUDITED.

## Current delivery status (2026-11-28)

Draft PR #125 now includes generator checkpoint `79bb57a`, including the CLI and
recent typed-parent, structural topology, policy, dual-target and runnable example
work. The latest completed full `make check` remains exact source `ce859899`:
5,214 passed, one skipped, one warning. That result excludes subsequent changes;
current-head full verification and independent review of newer deltas are pending.
Earlier checkpoint descriptions below remain historical component evidence.

Several old backlog statements are superseded by implemented paths:

| Previously listed gap | Current implementation and limit |
| --- | --- |
| Finite GP features inside DAGs | `GPRootDAGRecipe` / `generate_gp_root_dag`; retained finite populations, calibration/query envelopes and explicit interventions. See [GP populations](issue-28-gp-column-populations.md). Advanced automatic population policies remain #126. |
| Row-owned recursive mechanisms inside DAGs | `RecursivePopulationGraphRequest` / `realize_recursive_population_graph`; four authored graph families and retained comparisons. See [population DAGs](issue-28-recursive-population-dags.md) for lifecycle restrictions and historical repairs. |
| XGBoost ancestor/support/batch lifecycles | `fit_xgboost_population_graph` implements independent-ancestor, support-only and batch-reference fits; [actual-population contract](issue-28-xgboost-population.md). This is not every source/backend parameter option. |
| Typed topology and parent composition | `TypedParentTopologyRequest` supports four families, explicit/redirection topology and Cauchy/attachment/motif source composition; [contract](issue-28-typed-parent-topology.md). |
| Direct dataset creation without the explorer | `r0 generate --config REQUEST --out DIRECTORY` uses the guarded generation runner and persisted task groups; see the README. |
| Dual targets, spatial and high-frequency mappings | [Explicit component mappings and runnable examples](issue-28-atomic-delivery.md#dual-target-table); source-specific undisclosed algorithms are not claimed as reproduced. |

All UI/explorer delivery and the explicitly listed advanced compositions have moved
to [#126](https://github.com/Avistian/r0/issues/126) under the
[scope amendment](specs/0028-causal-network-generators.md#scope-amendment-generators-first-2026-11-28).
Older UI, browser, interactive-Colab and automatic-policy backlog statements below
must be read with that amendment. Required core source algorithms, numerical
inspection and explicit intervention semantics remain in #28.

The full option-by-option audit is still incomplete. The finite inventory contains
406 `REQUIRED_28` entries, 140 specification anchors, and separately classified
shared/interface/outside/undisclosed entries. These are tracking records, not 546
unique algorithms. Neither the presence of a module nor a passing example proves
all options in a broad group. Remaining work includes reconciling each in-scope
entry with actual behavior/evidence, reviewing newer deltas, current-head full
checks, and outstanding diversity/environment qualifications. Resource measurements
now include bounded 100-class cases for all four representations; the larger typed
cases rejected under the tested cap remain explicitly unqualified. See the
[resource report](issue-28-network-resource-measurements.md).

## Observations and network structure

| Inventory ID | Delivered component | Remaining qualification |
| --- | --- | --- |
| P1.graph.02 | Sampled observed network channels | P1 profiles are limited to their declared eligible layered catalog. |
| P1.graph.03 | Internal target selection | Target channels and feature exclusions are retained, not inferred from a root-input view. |
| P1.graph.04 | Final-layer target selection | Final-target overlap aliases are explicit in the pinned-code profile. |
| P1.graph.05 | Hidden unobserved channels | Valid disconnected/hidden states are preserved; no learnability pruning is implied. |
| P1.graph.06 | Contiguous feature windows | Paper-block and pinned-code restrictions remain separate named profiles. |
| P1.graph.07 | Unrestricted feature selection | The pinned-code final flattened index exclusion is retained. |
| P1.graph.08 | Preserved channel order | Ordered readout references survive archives and replay. |
| P1.graph.09 | Permuted channel order | Full selection permutation and raw construction remain retained. |

See [network observations](issue-28-network-observations.md) and the
[published observation checkpoint](issue-28-observation-checkpoint.md).
The explicit r0 propagated-block profile additionally supports variable-width,
multiparent, skip and gated architectures. It does not yet expose every expert,
gate or internal substage as a separate observation-catalog item. The broader
workbench integration is later than the published checkpoint.

## Recursive forests and population ownership

[MITRA A.2.2 and Algorithms 3–5](https://arxiv.org/html/2510.21204v1#A1.SS2.SSS2)
provide the recursive-tree reference. The implementation retains separate
random-start and zero-start class conventions and frozen versus per-row lifecycles.

| Inventory ID | Delivered component | Remaining qualification |
| --- | --- | --- |
| M1.tree.09 | Recursive path-conditioned split intervals | Path-local sibling copies are a named reconstruction. |
| M1.tree.10 | Minimum depth | Terminal draws are retained even when depth overrides the decision. |
| M1.tree.11 | Maximum depth | Construction admits the complete worst-case tree geometry. |
| M1.tree.12 | Early-terminal probability | Explicit configured probability; an undisclosed hyperprior is not invented. |
| M1.tree.13 | Random-start cyclic class leaves | Separate from Algorithm 4's zero-start convention. |
| M1.tree.14 | Zero-start cyclic class leaves | The otherwise-unused sampled starting value remains in custody. |
| M1.tree.15 | Gaussian regression leaves | Vector leaves are explicitly an r0 extension. |
| M1.tree.16 | Forest majority vote | Native labels and declared tie policies; winner indicators are not calibrated probabilities. |
| M1.tree.17 | Forest regression mean | Literal tree values are averaged with preserved feature order. |
| M1.tree.18 | Frozen forest shared across rows | Core is in the published checkpoint; shared graph adapters are later candidates. |
| M1.tree.19 | Independent row-owned forests | Terminal population envelopes are implemented and reviewed; internal DAG populations, descendants and complete workbench delivery remain in progress. |

Contracts: [recursive core](issue-28-mitra-recursive-forests.md),
[shared graph adapters](issue-28-recursive-forest-graphs.md),
[row-owned populations](issue-28-recursive-populations.md). A fresh forest per row
must never be archived as one frozen function. Changing rows on a realized
population requires a new construction request; matched comparisons retain each
row's original forest. Dynamic target calibration remains separate work.

## Feature convolution and hybrid mechanisms

`OP.function.07` has an actual feature-axis shared-kernel implementation motivated
by [O'Prior §2.1](https://arxiv.org/html/2605.18971v1#S2.SS1). The configurable
kernel, stride, dilation, padding, layout, initialization, activation, grouping,
depthwise and residual choices are explicit r0 constructions. This is not the
row-axis mechanism assigned to #39.

The core is in the published checkpoint. Later candidates include functional,
native, vector and converted graph use, parent/joint composition, sampled raw
custody and a guarded workbench with 17 presets. The mixed convolution/XGBoost
draft repair has independent browser and exported-value evidence. Typed reference
curves, internal innovation/importance compositions, broader source profiles and
complete workbench composition remain required. Sampled-convolution soft-policy
replacements at `6bb5ecd`, with shared-draw rendering repair `1a120893`, now have
bounded independent arithmetic/archive and browser acceptance; see their
[retained replacement contract](issue-28-sampled-convolution-policies.md) and the
[convolution contract](issue-28-feature-convolution.md).

`OP.function.08`–`.11` remain distinct: finite joint GP column draws followed by
linear, quadratic or neural target mechanisms. The finite-column core and strict
outer-envelope repair now have bounded independent acceptance at `5e87d34`,
integrated at `023a487`: shared/per-column RBF factors, all three target families,
independent calibration, matched/fresh target comparisons and complete replay.
The [GP workbench](issue-28-gp-workbench.md) is under development; hybrid DAG
roots, broader controls and full workflow qualification remain required. Existing
frozen GP functions and standalone GP target draws are separate mechanisms.
`OP.selection.01`–`.05` remain the #30 interface; selection ownership must not
silently migrate into this structural-mechanism slice.

## Sampled interventions

[TAFFY Appendix A.3](https://arxiv.org/html/2610.07559v1#A1.SS3) supplies the
scalar selection and intervention reference. The existing scalar profile covers
TA.intervention.01–.09 with a retained original graph, observed/hidden non-label
selection, constant hard replacements, explicit reference-population bounds,
soft numerical laws, fresh populations and the five-percent count rule.
The independent-calibration and matched-unit alternatives are named r0 extensions;
the four explicit soft parameter laws do not claim an undisclosed source prior.

See [sampled policies](issue-28-sampled-intervention-policies.md) and
[the workbench](issue-28-intervention-policy-workbench.md). Scalar nested-budget
and comparison-custody repairs have bounded acceptance and are included in the
completed `6c0cd82` suite.

The [typed-policy core](issue-28-typed-intervention-policies.md) adds numerical,
native logical-variable and named-capture addresses, declared categorical laws,
and retained sampled/fitted envelopes. Original `6c6770` and first repair
`eefdd77` remain historical **REJECT** versions for resource admission ordering.
The complete metadata planner at `935e10a` now has bounded independent acceptance:
13 caller-cap cases reject before replay or preserve exact valid receipts, and
110 literal numerical cases cover typed category, converter and convolution
compositions. This is later than the full-suite checkpoint. Typed workbench,
population/support refitting and remaining compositions are still required;
no complete typed-policy release acceptance follows from this mapping.

## Coverage beyond these components

The requested variety remains broad: source-specific topology and parameter
catalogs; richer observation/component assignments; root/prototype normalization,
internal noise and sampled profiles; XGBoost ancestor/support/batch lifecycles;
full retained graph/population combinations; and complete local/Colab creation,
editing, inspection, save and replay. Measured maxima, sustained throughput,
across-seed diversity, GPU/hosted execution and scientific qualification remain
separate open evidence requirements. None is replaced by preset or test counts.
