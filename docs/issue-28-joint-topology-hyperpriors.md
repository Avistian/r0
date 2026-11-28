# Joint mechanisms on topology hyperpriors

`JointHyperpriorGraphRequest` samples the complete existing source mechanism
catalog on a random-size forest. It is distinct from the explicit shared-template
[hyperprior graph](issue-28-hyperprior-graphs.md). The topology law retains the
[source-qualified interpretation and r0 choices](issue-28-topology-hyperpriors.md).
No forest-density claim or full P2 reproduction is made.

`sample_joint_hyperprior_graph` requires five distinct caller generators:
`topology_rng`, `named_rng`, `raw_rng`, `calibration_rng`, and `fit_rng`. The
independent topology stream draws N/P and every parent. The original joint graph
sampler consumes the remaining four, retaining its named shared scopes, raw
function draws, independent ancestor populations, fit receipts and snapshots.
All five publish atomically after successful binding. A numerical failure is
reported; no graph is retried or its catalog filtered.

The request exposes all current `JointFunctionPrior` catalogs, including ordinary
and Kumo product variants and mixtures; all five matrix families; all four
`JointRootPrior` base families; numerical preference modes `meta`, `local`, and
`global`; categorical `shared`/`refresh`; both global policies; structural noise
and a shared authored numerical pipeline. Width is explicit and uniform. Feature
and target references use the same explicit prefix/suffix layout as the authored
hyperprior envelope. Joint categorical names retain their existing contracts:
conflicting root/edge choice sets for the same name are rejected before any draw.
The separately named `r0-linear-projection-v1` root selector is available when an
explicit linear root and a different nonroot catalog are desired. Matrix names
still share their original scope and must have compatible catalogs.

## Resource contract

Static support propagation runs before copying a caller RNG. It includes every
reachable branch, not the branch later selected:

| Branch | Geometry support used |
| --- | --- |
| Tree | up to 128 trees, depth 7, all leaf output channels |
| MLP | one to 3 affine stages, hidden width at most 127, both endpoint activations |
| Discretization | explicit count or cardinality/fit-row bound |
| EM | at most max(16, 2×output width) centers |
| GP | explicit Fourier count, product and radial variants |
| Matrix | Gaussian, importance, factorized, kernel, transformed→simple |
| Kernel points | up to 15 mixture components in 3 dimensions |
| Product | two eligible non-product children, source-specific eligibility |

Counts include named numerical/categorical metadata, nested categorical weight
hyperparameters, complete registry copies, raw/derived function records, ancestor
prefix draws, graph evaluation and source-calibration floors. The source's
explicit bounded input truncation budget is reserved when selected; this does
not introduce a topology retry. Named preference and registry hard limits are
also checked. Complete five-stream state sizes are checked before cloning.

The named-record allowance follows the retained schemas: a numerical request,
preference, draw and event need fewer than 16 fixed numeric leaves plus 3 per
value; a categorical draw includes its two numerical weight-hyperparameter
records and at most 2 numeric leaves per choice. The reservation uses 64 per call
and value, 16 per choice and 512 per node to include plan/owner/span metadata.
String/container storage is charged separately. Raw records reserve 128 bytes
per numeric leaf and 16 KiB per header, above the current 96-byte leaf charge and
bounded owner/shape/law fields. This table is tied to the existing versioned
catalogs; adding a larger source branch requires updating the support planner.

These conservative reservations can reject broad priors under the ordinary
256 MiB/one-million-parameter defaults even when a particular random realization
would be smaller. A caller may explicitly authorize larger `CausalLimits` and
`RawDrawLimits`; these are admission ceilings, not immediate allocations or
measured RSS. No silent clipping, source removal or opportunistic retry makes an
inadmissible request succeed. The actual child sampler retains its own checks.

`JointHyperpriorGraph` stores both complete construction receipts. Reopening
validates outer controls, topology metadata, exact child request binding and
snapshot headers before restoring streams, then compares complete replay.
`generate_task`, task ZIPs, the ordinary guarded generation runner and
`explain_generation` retain this envelope. Original-row traces verify topology
and mechanism receipt fingerprints. `with_task` retains the topology and uses
explicit deterministic mechanism replay for the new task controls; its fit
population size remains the declared independent calibration size. Query rows
and labels never fit these mechanisms.

## Remaining breadth

Dedicated workbench construction/inspection controls, sampled observation and
width geometry, native/converted envelopes, parent composition priors and causal
intervention wrappers remain required follow-ons. Existing authored-template
support for frozen mechanisms remains available separately. This slice does not
claim those combinations, scientific equivalence, hosted/GPU verification, or a
full-suite result.

## Source-owned registry contract repair

Independent review of `6fe0a40` found that changing only the retained registry's
operation cap from 100000 to 99999 restored all five streams before complete
replay rejected it. The source sampler has no request-level override for this
cap. The original REJECT packet remains unchanged in that frozen worktree.

The joint-envelope preflight now binds registry and preference limits to the
sampler's actual defaults, outer mode/reuse policies to the graph request, the
operation root to `graph`, and nested categorical weight hyper-samplers to their
fixed meta/strict/default-limit settings. The generic configurable registry API
is unchanged. Validation discards no original records and performs no stream
restore or sampling. Existing ownership, raw-chain, snapshot, realization and
runtime checks still follow.

Public regressions retain the original two registry-cap failures and six analogous
preference-cap failures, all now rejected before restore. Other source-policy and
operation-root checks cover the complete contract. The outside original probe
reports zero restores for the invalid registry cap and five for its valid control.
Sixty complete serialized recipe/task artifact hashes match frozen `6fe0a40` across
all five engines and mode/policy combinations. The focused source-contract,
envelope, catalog and task-semantics checks pass; separate literal arithmetic/ZIP
and stream-rollback evidence is retained under ignored
`docs/evidence/issue-28-joint-source-contract/`. Independent repair acceptance and
full issue completion remain separate gates; no full suite was run here.
