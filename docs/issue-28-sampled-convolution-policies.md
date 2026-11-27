# Sampled convolution mechanisms in soft intervention policies

This extends `TA.intervention.01–09` / `R0.extension.21/.22` policy composition;
[the retained policy contract](issue-28-sampled-intervention-policies.md) supplies
the source selection and explicit r0 parameter-law distinction. It does not claim
that TAFFY prescribes convolution initializers or these four replacement laws.

Both scalar and typed policies can resample parameters of a sampled Conv1D
mechanism, directly or within a `FrozenParentFunction` child or joint member.
A law addresses the original record, for example
`("function", "layers", "0", "kernel")` or
`("children", "0", "function", "layers", "0", "bias")`.
Only the realized function is an eligible parameter scope: initializer request,
raw draws and RNG snapshots are not resampleable parameter groups.

Normal, uniform, log-uniform and choice laws retain their existing explicit
`*-r0` names and transforms. The original graph and complete sampled kernel
receipt remain in `request.base`. Only affected sampled Conv wrappers become
`FrozenFeatureConv1D` authored replacements. Unchanged sampled siblings keep
original receipts and strict replay validation. This changes provenance wrapper,
not equation family, dimensions, convolution geometry, parents, activation
selection, unlisted coefficients or exogenous laws. No fabricated initializer
receipt is attached to replacement weights. No retries or clipping occur.

Functional, native categorical, native-vector and converted graph runtimes use
this same replacement seam. Matched/fresh policy modes retain their existing
semantics; categorical emission and converted captures continue after the changed
numerical equation. Architecture, family, integer indices and category vocabularies
are not continuous parameter laws. Source-profile eligibility restrictions remain.

Archives retain both original construction and replacement draws and replay the
complete policy before generation. Original-row traces expose historical kernel
construction nodes separately from replacement policy draws. Their metadata links
the original path and replaced paths and explicitly says original initializer
draws did not produce overwritten coefficients. Active convolution stages use the
authored replacement. Existing valid records/tensors are unchanged; trace metadata
adds original-construction references (empty for non-sampled mechanisms).

Workbench controls for composing these paths and broader fitted/backend mechanism
resampling remain separate required work. No hosted or whole-issue acceptance is
claimed by this library composition slice.
