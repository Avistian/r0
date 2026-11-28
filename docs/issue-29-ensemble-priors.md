# Sampled ensemble size and weight priors

The existing `tree-ensemble` construction accepts an optional `prior` serialized
from `r0.modules.generation.TreeEnsemblePrior`. Without it, the authored member
list, weights, random streams and v1 receipt remain unchanged. With it, `members`
is a bounded template pool and authored `weights` must be empty. Task and root
controls must agree across all templates; individual tree structure/leaf draws
remain independent by selected member slot, even when a template is selected
more than once.

This is an r0 extension for T29-05/06/07, not an assertion that a paper checkpoint
used these distributions. Use
[`prior-tree-ensemble-cpu.json`](../configs/generation/prior-tree-ensemble-cpu.json)
for a small API/CLI example. The configuration is inspectable JSON containing
complete template requests and the prior, not a hidden preset.

## Distributions and support

| Control | Semantics |
| --- | --- |
| `min_members`, `max_members` | Inclusive integer count support, bounded within 1..128. |
| `count_probabilities` | Optional explicit categorical probabilities for every count in that interval. Empty means discrete uniform. Zero-probability counts remain declared but are not drawn. |
| `member_probabilities` | Optional categorical probabilities over the ordered template pool. Empty means uniform. |
| `replacement` | With replacement permits repeated templates; without replacement samples a weighted ordered subset and requires enough positive-probability templates for the declared maximum count. |
| `weight_law="equal"` | All realized raw weights are one. |
| `weight_law="dirichlet"` | Symmetric Dirichlet weights with per-component `concentration` in [0.05,100]. This is per-component concentration, not fixed total concentration across sizes. |
| `weight_law="lognormal"` | Independent weights with log mean zero and log standard deviation `scale` in (0,4]. |
| `weight_law="signed-normal"` | Independent zero-mean Normal weights with standard deviation `scale`; only sum reduction is allowed. |

Probability arrays must have the required length, finite entries in [0,1] and
sum to one within an absolute tolerance of 1e-12. They are not silently normalized.
`concentration` affects Dirichlet only; `scale` affects lognormal/signed-normal
only. Both remain in the versioned request for replay. Positive weight laws can
use mean or sum. Mean divides by the realized raw weight sum; sum retains raw
magnitudes. Signed-normal mean is rejected because near-canceling denominators
would introduce an undeclared amplification regime. A normal draw need not
contain both signs in every realization.

The sampler uses a dedicated PCG64/SeedSequence namespace with construction seed,
episode seed and task ID. It draws count, ordered template choices, then weights.
Member trees retain their existing separate slot-indexed streams. A v2 construction
receipt stores the full prior, selected template indices, realized weights,
reduction and every member's complete sampled-tree receipt. Saved task matching
reconstructs this receipt, so changing a prior or draw breaks resume identity.
No mutable process-global RNG is used.

## Bounds, verification and remaining work

All templates are parsed and checked for compatible task/root controls before
sampling. The preflight node bound uses declared maximum count × the largest
candidate template bound, plus ensemble overhead. This is conservative, including
zero-probability templates/counts; it does not claim a tight expected-cost bound.
The compiler applies its remaining channel/parameter/working-memory limits.
Invalid policies and node budgets fail before size/selection/weight draws.

Focused checks reconstruct every member and independently combine its target
values using recorded weights for all four laws. They verify exact repeated
construction identity, categorical support, independently drawn repeated templates,
preflight ordering, saved archives and guarded resume. Existing authored ensemble
checks run alongside them. A CLI pilot and explicit saved numeric aggregation
are retained under ignored `docs/evidence/issue-29-ensemble-priors/`.

These priors do not yet implement correlated tree structures/leaf parameters,
bootstrap-like shared fitting populations, probability-leaf/vote policy breadth
or an exhaustive source-specific forest audit. Reusing a template is not the
same as sampling a deliberately correlated forest. Full #29 remains open.
