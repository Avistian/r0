# Per-parent function composition

This increment implements a complete opt-in `sample_parent_graph` construction
on an explicit `JointGraphRequest` DAG. It preserves the existing joint-concat
sampler and its archive schema. `ParentCompositionPrior` selects the source
contract. `SampledParentGraph` retains four initial/final RNG snapshots, every
named decision and typed owner, original raw draws, ancestral-fit receipts,
child realizations, aggregate admission, and executable frozen equations.
Reopening and validation replay the entire construction and compare every field.

## Source contracts

The pinned [TabICLv2 MultiFunction source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_multi_function.py)
defines joint concatenation or separate parent functions. The latter maps each
parent to the output width and reduces the member axis using sum, product,
maximum, or logsumexp. The source parameter names `multi_function_func_type`
and `aggregation_op_type`, ordered catalogs, and shared sampling context are
retained. `mode="concat"` and `mode="aggregate"` bypass the mode selector;
`aggregation` can explicitly fix one admissible operator. The public aggregate
spelling maps to the upstream `agg` configuration spelling.

[MITRA2 section 2.3](https://arxiv.org/html/2609.04540v1#S2.SS3) uses a joint
nonlinear output and independent linear projections of each parent for nodes
with multiple parents. Their outputs are **members of one reduction**, using
mean, softmax-weighted sum, MLP projection, product, or maximum. The joint
output is not added after the reduction. Single-parent nodes use their joint
function directly.

MITRA2's exact projection, weight, and neural parameter laws are not fully
specified by that passage. This implementation explicitly adapts the existing
r0/TabICLv2 sampled function/matrix machinery: each projection is a sampled
linear function; the weighted reducer freezes softmax of independent standard
Gaussian logits (`r0-softmax-gaussian-logits-v1`); the neural reducer is the
existing sampled MLP applied to member-major concatenation. Forced projection
and reducer catalogs have distinct versioned names
`r0-linear-projection-v1` / `r0-neural-reducer-v1`, so they cannot change the
shared ordinary function catalog's contract. These are r0 parameter-law
adaptations, not claims of exact MITRA2 reproduction.

The `extension` source exposes separate parent functions with the union of all
seven reductions. Mean, weighted, and neural reductions are **not** attributed
to the TabICLv2 source catalog. Both root and ordinary child function priors
continue to obey the existing shared-name catalog contract.

## Equation and calibration boundaries

`FrozenParentFunction` contains one nonrecursive frozen function per parent.
Authored members and the optional joint member also accept immutable fitted-tree
functions; their exported split tables run without refitting. The sampled parent
selector still draws from its original eight families. Their input widths define disjoint consecutive slices in parent order;
graph admission checks every slice against the corresponding parent's width.
An optional joint function is the first reduction member. All members share the
output width. Products use conservative dependent-moment bounds; no claim of
parent independence is inferred from a DAG or finite calibration population.
Nonfinite results fail without normalization, replacement, or resampling.

Every fitted child obtains a fresh ancestral population through already
finalized parent equations, including authored pipelines and structural noise.
The child retains only its input slice in its ordinary fitted realization. The
receipt points to the original raw ancestor draws for the whole population;
complete replay recovers and checks the slice. A neural reducer gets another
fresh ancestor population transformed through the frozen member functions.
None of these populations contains emitted task rows. All raw, fit, calibration,
and named RNG state advances publish atomically after complete admission and
runtime validation. Parameter/memory bounds sum all child construction costs,
retained records, prefix states, member buffers, and ledger/replay copies.

Generation, recipe decoding, inspection, source identity, split changes, and
paired interventions accept the new version. Inspection exposes each parent
transform, the joint member when present, and the exact final reducer.

## Deliberate remaining work

This is an explicit-topology construction increment. Sampling geometry/layout,
noise policies, numerical pipelines, native-family topology, UI controls, and
Colab/local creation transport for parent composition require follow-on
integration. Existing authored pipeline/noise settings work in this constructor;
this is not a claim that their full sampled option families are wired here.
Fitted structural tree members are admitted for authored compositions; sampling
their fit plans inside the shared parent selector remains follow-on work. Broad #28 and paper-coverage acceptance remain open.

## Scoped validation

The focused parent suite covers all seven reducers with independent NumPy
arithmetic, all eight child function families, sixteen source/extension reducer
cases, replay and archive tampering, split identity, paired noise changes,
ancestor pipelines/noise, strict per-parent widths, dependent heavy-tail moments,
nonfinite failure, and transaction rollback. Thirty-five cases passed. A broader
frozen-function/functional-graph subset passed 69 cases before the seven added
extension cases; the parent plus legacy archive/plan subset passed 227 cases.
These are scoped checks, not full-suite acceptance.

An independent base-code invocation at `c9eed2e` and this implementation produced
identical SHA-256 hashes for one deterministic legacy sampled graph archive,
generated recipe JSON, and mechanism JSON. This checks a retained legacy path;
it is not an exhaustive byte-identity proof for every possible legacy input.
Raw logs and the hash-reproduction script live only under the ignored
`docs/evidence/issue-28-parent-composition/` directory.
