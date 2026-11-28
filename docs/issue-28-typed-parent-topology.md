# Typed topology with parent composition

`TypedParentTopologyRequest` adds a separate, opt-in request version to typed
geometry. It accepts the existing fixed-size or random-size redirection hyperprior,
or `FixedParentTopology` with explicit ordered parent indices. An explicit topology
retains every edge and consumes no topology RNG draws. Redirection forests still
have at most one parent per node; multi-parent reductions require an appropriate
explicit graph. No extra topology is sampled by the geometry or mechanism adapter.

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

Remaining qualification includes independent review, a neural-reducer numerical
oracle, broader random-topology composition, complete workbench controls and hosted
Colab delivery. Full-suite acceptance is separate from focused and archive checks.
