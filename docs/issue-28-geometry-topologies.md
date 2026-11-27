# Structural topology integration with sampled geometry

`GeometryGraphRequest.topology` accepts three explicitly versioned frozen priors:

| Prior version | Public controls | Meaning |
| --- | --- | --- |
| `cauchy-graph-prior-v1` | `CauchyGraphPrior(nodes, offset)` | Existing Cauchy-logistic forward topology |
| `degree-attachment-dag-v1` | `AttachmentGraphPrior(nodes, components, edges_per_node, attractiveness, power)` | Degree-attachment r0 extension |
| `recursive-motif-dag-v1` | `MotifGraphPrior(depth, components, motifs, rewrite_steps, rewrite_kinds)` | Hierarchical motif and acyclic rewrite adaptations |

The structural variants retain the contracts in
[structural topologies](issue-28-structural-topologies.md). Their source/adaptation
labels are unchanged. All motif choices and rewrite catalogs are preserved;
configuration decoding rejects unknown versions and extra fields. A motif prior
serializes depth/components; its node count is computed, not a second input.

No fields were added to the existing geometry envelope or Cauchy record. The
original Cauchy sampling call, draw order and aggregate limits are preserved.
Structural variants use the same complete geometry construction, shared named
preference context and four caller streams as Cauchy. Their `StructuralGraph`
record, all raw choices and rewrite receipts remain inside the saved recipe and
mechanism card. Reopening reconstructs the whole construction before comparing
all retained fields. Empty topology transcripts are valid for isolated attachment
roots and depth-zero motifs; raw continuity starts at the initial snapshot and
continues through observation and mechanism draws.

Shared generation, original-row inspection, task resplitting and supported paired
task-control comparisons now operate on these variants. This does not turn task
noise/separation comparisons into hard or soft causal interventions. Binary and
multiclass cases use the existing probabilistic adapters, not native categorical
SCM variables.

## Combined raw resource accounting

Topology and observation costs are reserved before any caller stream advances.
Mechanisms receive the remainder. Structural sampling also receives the caller's
raw caps; it does not reset them to defaults.

For each attachment component with `s` nodes, predecessor count `k` runs from
1 through `s-1`; let `q = min(edges_per_node, k)`. The exact call count is
`sum(q)`, with one scalar emitted per call. The number of retained categorical
probabilities is `sum(q*k - q*(q-1)/2)`. The implementation evaluates these sums
in closed form for the two possible balanced component sizes. It does not
allocate or iterate through an unbounded prior merely to calculate a budget.

Motifs reserve `components*(3**depth-1)/2` expansion calls and at most three calls
per requested rewrite. Each emits one scalar. A no-op may use fewer calls, but
the conservative reservation is not reclaimed or used to resample the graph.
Cauchy keeps its existing four calls and `nodes**2 + 2*nodes + 1` values.

Together with observation draws, retained raw storage reserves 96 bytes per
emitted value, 4096 bytes per record, 128 additional bytes per attachment
probability, and four configured state-byte allowances. The last term covers
topology and observation transactions. Existing geometry/mechanism metadata
admission remains enforced separately. These are admission estimates, not peak
RSS measurements. Budget failures preserve all four original caller states.

## Retained evidence

Author-run evidence lives under ignored
`docs/evidence/issue-28-geometry-topologies/` in
`/tmp/r0-28-geometry-topologies`.

- `red.log`: eight failures from the previously Cauchy-only request contract.
- `green.log`: initial eight end-to-end/aggregate-budget cases passing.
- `engines-adapters.log`: 23 cases passing, including five engines, classification
  adapters, declared larger limits and tampering across an empty topology prefix.
- `compatibility.log`: 42 passing structural integration and existing Cauchy
  geometry cases, including exact attachment support counts and no-op rewrite
  upper-bound reservations.
- `static.log`: initial typing failure for an untyped JSON mapping;
  `static-final.log`: repaired static checks; `static-precommit.log` repeats them
  after adding reservation boundary cases.
- `cauchy_compatibility.py`: retained reproduction program for the before/after
  canonical archive comparison at base `16bfd2319b23f5884810f7bae4695e94d9faa5e9`.

The before/after Cauchy archive is 45,155 bytes, with matching SHA256
`dd4200dc00ade9449a55109eb3fbc970302d667e8794b25c6c0bf60aca79fd27`.
The remaining raw limits are unchanged: 99,992 calls, 999,975 scalar values,
264,205,984 working bytes and 1,048,576 state bytes. Exact JSON bytes are retained
in `cauchy-before.json` and `cauchy-after.json`, with their receipts in the matching
`.log` files. Reproduce the comparison with the retained program at each head,
then compare the resulting files byte-for-byte.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_geometry_structural_topologies.py \
  tests/priors/test_sampled_geometry_graph.py
```

Independent review and the parent-coordinated full suite remain required.
Explorer controls, browser/Colab demonstrations, causal interventions, measured
resources and structural-diversity qualification retain their separate pending
requirements. This integration does not close full issue #28.
