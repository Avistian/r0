# Typed topology with parent composition

`TypedParentTopologyRequest` adds a separate, opt-in request version to typed
geometry. It accepts the existing fixed-size or random-size redirection hyperprior,
or `FixedParentTopology` with explicit ordered parent indices.
`StructuralTopologyRequest(prior)` also composes the existing `CauchyGraphPrior`,
`AttachmentGraphPrior` and `MotifGraphPrior` with all four typed families. An explicit topology
retains every edge and consumes no topology RNG draws. Redirection forests still
have at most one parent per node; multi-parent reductions require an appropriate
explicit graph or a multi-parent structural topology law. No extra topology is sampled by the geometry or mechanism adapter.

The four mechanism families use their existing parent samplers: functional geometry,
native scalar, native vector and converted geometry. Their role laws, width semantics,
converter vocabularies, numerical pipelines and named catalogs remain available.
The version is `typed-parent-topology-request-r0-v1`; its saved envelope is
`sampled-typed-parent-topology-r0-v1`. Existing typed v1 records keep their schema.

| Source selection | Composition | Reductions |
| --- | --- | --- |
| `tabiclv2` | Concatenation or separate parent functions | sum, product, max, logsumexp |
| `mitra2` | One parent: concatenation. Multiple parents: joint nonlinear member plus one linear projection per parent | mean, weighted, mlp, product, max |
| `extension` | Concatenation or separate parent functions | sum, product, mean, max, logsumexp, weighted, mlp |

The source labels reuse the existing documented operator interpretations. Numerical
parameter laws and the explicit topology/typed-geometry composition are r0 adaptations,
not claims of complete paper reproduction. A caller can retain a fixed reduction or
sample the full source-specific catalog. MITRA2 retains its joint member before
reduction; it is not added after reducing the separate parent projections.

Admission reserves the maximum physical channel width, concatenated parent inputs,
all simultaneous members, and neural-reducer support even when edge functions are
linear. The bound is conservative and may require larger caller caps than a realized
small graph. It is not a measured memory or throughput claim. Construction failures
roll back all five caller streams. Reopening validates request, topology, child and
stream ownership, then compares a complete reconstruction. Original-row traces
identify explicit topology as retained rather than sampled.

Structural topology records retain their complete original source graph and dedicated
stream snapshots. They keep original attachment probabilities, Cauchy proposals and
motif rewrite receipts; no connectivity retry is added. Admission reserves maximum
fan-in and the topology transcript before mechanisms are sampled. Source cards name
the structural law rather than attributing it to P2 redirection.

Remaining generator qualification includes independent review and full coverage/resource
audits. UI/workbench and interactive Colab delivery belong to #126 and no longer block #28. Independent
NumPy fixtures now check sampled neural reducers, including calibrated activation
stages; this does not establish coverage of every activation/reducer combination. Full-suite acceptance is separate from focused and archive checks.
