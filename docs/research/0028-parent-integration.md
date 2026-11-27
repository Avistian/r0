# Parent composition across construction paths

This follow-on connects the source contracts in
[the parent-composition note](0028-parent-composition.md) to sampled numerical
geometry, independently fitted numerical pipelines, and sampled native variable
domains. It adds no new paper-fidelity claim: the TabICLv2 catalog and the
MITRA2 joint-member semantics retain the source/adaptation distinctions in that
note.

## Public controls

`sample_geometry_graph(GeometryParentRequest(...))` samples topology, observation
assignment, converter channels, channel widths, parent mechanisms, and optional
node pipelines in one shared named context. `composition` is a
`ParentCompositionPrior`. `pipeline_prior=None` keeps the existing authored
numerical readout/noise settings; a `JointPipelinePrior` selects independently
fitted numerical pipelines and their sampled noise. All existing sampled
geometry topology choices remain available. The request has version
`geometry-parent-request-v1`; the stable geometry envelope contains a
`MaterializedParentGraph` with version `materialized-parent-graph-v1`.

`sample_native_graph(NativeParentGraphRequest(...))` uses the same composition
control with sampled continuous, binary, or categorical domains. Each parent
function consumes its parent's **encoded state width**, including the complete
one-hot representation for discrete parents. Members and the optional joint
function produce the child's **score width**, before native emission. This
preserves the distinction between one observed categorical value and its
structural encoding. The request has version `native-parent-graph-request-v1`;
the stable native archive envelope retains additional parent-fit receipts.
Native topology is explicit, as in the existing native sampler. This does not
add a sampled native-topology/layout constructor.

Both new request types are opt-in. Old requests retain their original record
fields, sampling paths, and complete-replay behavior. Decode uses the nested
request version to reconstruct the correct complete archive subtype. Recipe
roundtrip, original-row inspection, zip group saving/loading, and episode
resplits work through the existing public interfaces. Numerical geometry also
supports the existing noise/separation pairs; native targets retain their
existing prohibition on label-adapter interventions.

## Fit and admission boundaries

Geometry construction plans all parent functions and pipelines before drawing
numerical arrays. Each fitted parent function receives a fresh population
through finalized ancestor equations, including their already fitted pipelines
and noise. A neural reducer gets another independent population transformed
through its frozen members. The receiver's pipeline is fitted only after its
parent reduction is frozen. All intervals refer to the original shared raw
ledger; replay reconstructs the exact sliced or transformed fit matrices.

Native fitting evaluates the finalized native prefix, including each discrete
emission and its retained calibration variates, before slicing encoded states.
Parent-fit receipts retain node, role, parents, and original raw interval. A
native-domain realization and a parent-mechanism realization share the same
named context and staged raw/calibration/fit streams.

Aggregate admission sums all member construction costs, neural reduction
buffers, ancestral state storage, raw ledger/replay budgets, and serialized
parent-reference receipts. Four caller streams publish only after successful
construction. Large sampled source mechanisms may fail these finite budgets;
they are not retried. One explicit tree draw rejected by the default native
budget is retained as a rollback test. A separate, explicitly chosen bounded
fixture exercises the successful fitted-tree path.

## Validation and remaining boundaries

Scoped tests cover geometry with and without pipelines, joint-member and
separate-parent semantics, native continuous/binary/categorical domains across
all seven reductions, all eight sampled function families, independent neural
reducer/pipeline ordering, replay/tampering, encoded widths, resplits, pairs,
zip roundtrips, and budget rollback. Legacy numerical pipeline, channel-span,
explicit-parent and native tests remain part of the scoped regression set.
Full-suite acceptance remains with the integration owner.

Three deterministic legacy fixtures (sampled numerical pipeline geometry,
sampled native graph, and the first explicit-parent archive) produced identical
archive, generated recipe, and mechanism JSON hashes when run against base
`259a496` and this implementation. These are targeted byte-identity probes,
not a proof covering every possible old input. Logs and reproduction scripts
are retained only under ignored `docs/evidence/issue-28-parent-integration/`.

UI/Colab construction controls are outside this worker's ownership. The
independent fitted structural-tree worker also owns its separate sampled
family selector; root integration adds authored fitted-tree parent-member
admission. No new fitted-tree selection law is implied by these eight-family
source sampler paths. Broader #28 options and end-to-end publication acceptance
remain open.

The final integration suite passed 38 cases; the selected legacy/explicit-parent
regression suite passed 78. The tests retain both an oversized native tree
rejection and a numerical pipeline's uncertified exponential-warp rejection,
checking complete RNG rollback. The positive neural-reducer/pipeline fixture
explicitly disables optional converter warps; it does not retry the rejected
source draw. Static type checking passed before the commit hook.
