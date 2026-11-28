# Structural topology candidates

These samplers add missing topology components from accepted decision 4. They do
not close the shared generation/workbench delivery requirement or whole issue #28.
They are distinct from the existing forward/layered/redirection sampler and the
Cauchy sampler; none of those earlier implementations is replaced.

## Source coverage and adaptation boundaries

| Source option | Component | Status and interpretation |
| --- | --- | --- |
| R0.extension.02 | `AttachmentGraphPrior` | Explicit degree-based attachment extension, distinct from redirection |
| L2.structure.04–06 | `MotifGraphPrior.motifs` | Chain, fork/common-cause and collider structures |
| L2.structure.07 | Recursive substitution | Multi-level local motifs with explicit input/output ports |
| L2.structure.08 | `redirect` rewrite | Replace an edge's source with an eligible earlier node |
| L2.structure.09 | `path` rewrite | Replace one edge with a two-edge path through an existing node |
| L2.structure.10 | `bypass` rewrite | Remove a through-node's incident edges and join its parents to children |

[LimiX-2 v1, §4.2](https://arxiv.org/html/2609.17488v1#S4.SS2) describes hierarchical
causal motifs and acyclic transformations, but does not specify their full
sampling distributions or transformation algorithms. The concrete variants here
are named r0 adaptations. They are not claimed to reproduce its private prior.

[Organization of Growing Random Networks v2](https://arxiv.org/html/cond-mat/0011094v2)
provides the degree-dependent attachment framework and distinguishes direct
attachment from redirection. Our configurable positive-attractiveness,
multiple-parent, forward-causal implementation is an extension, not an exact
reproduction of a published initial condition or prior mixture.

Sources were inspected as primary paper text. No upstream code was copied.
Existing source-option rows and their remaining integration/evidence obligations
are retained; sampler-level coverage is not option-level delivery acceptance.

## Preferential attachment contract

`AttachmentGraphPrior(nodes, components, edges_per_node, attractiveness, power)`
partitions nodes into contiguous components using integer proportional boundaries.
Each component begins with a root. A new node chooses up to `edges_per_node`
distinct predecessors. At each choice, candidate weights are proportional to
`(current undirected degree + attractiveness) ** power`. The selected edge
increments both endpoint degrees. Candidates already chosen for that new node
are removed. Edges point from earlier causes to later effects.

`attractiveness` must be positive; `power` is nonnegative. Power zero is uniform
attachment; powers below/at/above one expose sublinear/linear/superlinear kernels.
Normalized probabilities use log weights with a common maximum subtracted.
Nonfinite weight arithmetic is rejected; it is not silently clipped or retried.
The component root has degree zero until its first child. This explicit initial
condition is part of `degree-attachment-dag-v1`.

## Recursive motifs and rewrites

`MotifGraphPrior(depth, components, motifs, rewrite_steps, rewrite_kinds)` starts
with one node per initial component. Each expansion replaces every node with a
uniformly selected three-node motif. Chain edges are 0→1→2; fork edges are 0→1
and 0→2; collider edges are 0→2 and 1→2. Incoming ports are motif roots and outgoing
ports are motif sinks. An old edge becomes all sink-to-root connections between
the corresponding replacement blocks. Node order remains topological. Final
node count is `components * 3**depth`.

Each rewrite step first samples a kind uniformly from its declared catalog.
Redirection and path replacement select one existing edge uniformly. Redirection
selects a different absent source edge to the same target, within the initial
component and forward order. Path replacement selects an intermediate index
strictly between the edge endpoints, requiring both replacement edges absent.
Bypass selects a node uniformly; a node with both parents and children loses its
incident edges and gains all parent-to-child shortcuts. That node remains as an
isolated node; node identities and count are preserved.

No eligible edge/replacement/through-node is an explicit recorded no-op, never a
retry. Rewrites never cross the original component partition. They can create
additional disconnections. Valid disconnected, nuisance and common-cause
structures are not filtered for target predictability. Observation choices and
hidden-variable roles belong to the separate observation layer.

## Record and replay API

`sample_structural_graph(prior, rng=..., limits=..., raw_limits=...)` accepts an
explicit concrete NumPy generator. `StructuralGraph` retains the prior, ordered
parents, complete raw choice records, rewrite receipts and causal limits.
Receipts record subjects, removed/added edges, and disposition. Every categorical
probability vector is retained, not just its selected outcome.

`validate()` re-derives every dynamic choice contract, graph and receipt from the
raw transcript and rejects extra/missing/changed records. It checks structural
consistency, not historical RNG provenance. `replay(rng)` additionally verifies
all draws from the matching pre-state and publishes caller state only on success.
Construction and replay failures leave caller state unchanged.

Admission conservatively bounds full probability vectors, intermediate edges,
receipt copies and working metadata before construction. Raw-ledger caps apply
as well. These are analytical admission estimates, not measured peak RSS.
Large graph/depth/rewrite combinations may require explicit larger exploration
caps; limits never shorten the requested construction.

## Verification and outstanding delivery

Reproduce in `/tmp/r0-28-topology-options-impl` with the existing environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_structural_topologies.py
```

Evidence is ignored under `docs/evidence/issue-28-topology-options/`. `red.log`
retains missing-API failures. `green.log` retains the first five passing cases;
`engines-rewrites.log` records 19 passing cases with five-engine replay, independent
degree-weight sampling, exact motif ports, receipt reconstruction, explicit
kernel variants, disconnected cases, tampering, and raw-budget rollback.
`operation-oracles.log` repeats the 19 cases with exact operation-specific edge
delta oracles. `static.log` retains two test-style failures; `static-final.log` is the repaired
static run. Full-suite verification is coordinated by the parent agent.

Integration with shared construction archives, saved inspection, UI/Colab,
interventions, source-option artifacts, across-seed structural diversity and
measured resource evidence is still required. No downstream predictive advantage
or exact upstream distribution identity is claimed.
