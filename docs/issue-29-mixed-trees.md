# Mixed numeric and categorical model trees

`mixed_model_tree_recipe` compiles a tree whose branch feature indices select
either `NumericProjectionSplit` directions or `CategorySubsetSplit` predicates.
Numeric predicates compute a dot product. Categorical predicates compute exact
zero/one membership; their thresholds and intervals use that bit space. Leaf
models consume the original input columns. Arbitrary finite numeric directions,
category subsets, tree shapes and existing affine/activated leaf models can be
authored explicitly within the existing limits.

Equality is explicit: `left` uses `<=`, and `right` uses `<` for left routing.
The right-equality compiler negates predicate outputs and thresholds and swaps
children, including for categorical predicates. Empty subsets, unseen codes,
all-zero directions and inactive leaf models keep their existing semantics.
Nonfinite inactive-model evaluation still fails rather than being hidden.

## Per-task sampling and CLI

`TreeExtensionRequest.families` now accepts `mixed` and `mixed-pruned`. These
require an uncalibrated `SourceRootBase` with at least one numeric and one
categorical coordinate. Existing families and their RNG consumption are unchanged.
This is an r0 extension, not a claimed source-paper tree prior.

Each sampled branch independently chooses its predicate kind with probability
one half. Categorical branches choose an eligible column uniformly, then include
each vocabulary code with `split_density`; membership threshold is 0.5. Numeric
branches draw Gaussian coefficients over numeric columns only, mask coefficients
with `split_density`, and draw a uniform threshold within the configured bound.
Categorical columns have zero coefficients in these numeric directions. Both
predicate kinds may occur in one tree, but all-numeric or all-categorical outcomes
are retained without resampling. No unused temperature draws are made.

Existing balanced/pruned geometry and constant/linear/tanh/sine leaf sampling
apply unchanged. A depth-zero tree remains one model leaf. Pruned geometry retains
the documented depth-preserving left-spine law. The current sampler's kind
probability is fixed at one half; arbitrary authored mixed trees remain available.

```sh
r0 generate --config configs/generation/sampled-mixed-tree-cpu.json --out /tmp/mixed-trees
```

The existing construction receipt, JSON decoder, guarded generator, archive and
resume matcher retain/replay these choices. No new request field changes older
serialized request shapes. Subset predicates operate on numeric codes; native
observation schemas remain a separate explicit declaration/conversion.

## Evidence and remaining breadth

Thirty-nine focused checks pass across mixed trees, existing extension sampling
and categorical trees. They cover exact thresholds under both equality rules,
unseen category codes, balanced/pruned sampling, exclusion of categorical columns
from numeric directions, depth zero, JSON replay and dataset archives. Targeted
type checks and repository formatting pass.

The CLI example generated three 64-row datasets. An independent NumPy calculation
evaluated each retained predicate, traversed the saved recursive tree and selected
its raw-input leaf model. All three score arrays matched exactly in those examples;
two trees contained both predicate kinds and one contained only numeric branches.
The construction and output files, raw logs and oracle are retained under
gitignored `docs/evidence/issue-29-mixed-trees/`. These checks do not establish the
entire #29 portfolio, maximum resource combinations or downstream predictive gain.

Correlated forests, sampled ensemble size/weight laws, probability/output options,
broader source profiles and diversity qualification remain separate work.
