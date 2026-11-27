# Sampled native graphs in the explorer

Choose **Construct a causal graph → Native mixed graph (sampled domains)** in the
local explorer or the shared notebook page. The editable JSON exposes the complete
`NativeGraphRequest` controls, except exploration limits, which use the existing
caller-owned **Exploration caps** field. The initial example has continuous,
binary and categorical observed variables and a categorical target on an explicit
DAG. It is an editable example, not a prescribed production mixture.

The construction seed samples domains and mechanisms using four retained RNG
streams. The dataset seed draws emitted rows. `rows`, `context_rows` and `split`
in the construction controls determine its episode geometry; target type and
class count follow the sampled target domain. The shared guarded generation
transport binds this row geometry before sampling, then saves the complete
sampled archive. Dataset shape fields above the construction panel do not override
these controls. Root and child function catalogs must have compatible shared
named preference contracts. Malformed row, context or observed-channel controls
are validated inside the guarded worker after the attempt receipt is created;
the failed attempt is retained and the active dataset stays unchanged.

Each `domains` entry exposes kinds, cardinality catalogs, emissions, frequency
laws, concentration/temperature bounds, threshold laws/scales and continuous width.
Uniform is an explicit baseline. The configurable symmetric Dirichlet hyperprior,
frequency offsets and reference cutpoints are adaptations; logistic-normal is an
extension. Frequencies influence logits or reference cutpoints, **not guaranteed
marginal class counts**. Independent sorted-normal cutpoints do not use frequency
weights. See [the sampled-native semantics](sampled-native-graphs.md) for the exact
laws and retained provenance. Targets retain the 100-class task limit; larger
feature vocabularies remain subject to the selected resource limits.

The view displays declared categories and direct native targets. Generation
inspection reconstructs selected original rows, including repeated indices.
Download retains typed CSV, schema/identity and replay data; query targets remain
absent from predictor CSV. Reopening verifies the sampled construction. Inspecting
the mechanism exposes its frozen native recipe; changing equations or domains
creates a linked authored edit while retaining the original sampled archive.
These edits are not causal interventions. The edited domain must remain coherent:
for example, changing softmax to argmax also requires temperature 1. Linked edits
can be saved, reopened and exported through the existing workbench.

Public lifecycle tests cover three frequency laws, typed exports, original-row
inspection, edits, reopening and failed-request state preservation. Separate
checks cover continuous/binary target derivation and transport geometry binding.
A real installed-Chromium walkthrough is reproducible with
`scripts/sampled_native_explorer_browser_smoke.mjs CDP_PORT EXPLORER_URL OUTPUT_DIR`;
run the explorer and Chromium on independently owned local ports first. Generated
logs, screenshots, receipts and downloads belong under ignored `docs/evidence/`.

Hosted Colab execution, exhaustive prior combinations and full-suite acceptance
remain separate verification. Heterogeneous variables per node,
further disclosed source laws, and native intervention-pair
APIs remain required broader #28 work. JSON controls do not complete those items.

Asymmetric Dirichlet is an explicit JSON opt-in: set a domain's `version` to
`native-asymmetric-domain-prior-v1`, select `asymmetric-dirichlet` in
`frequency_laws`, and supply `concentration_vectors` (one positive vector for
each reachable category count). Entries are absolute concentrations; the scalar
concentration interval is unused for this branch. This r0 extension supports
native parent constructions and the same typed inspection/export/reopen path.
See [the native law contract](sampled-native-graphs.md) for scope and provenance.
