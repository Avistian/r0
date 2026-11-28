# Category-subset model trees

`categorical_model_tree_recipe` compiles explicit category-subset splits and
constant/linear/activated score-vector leaves into a frozen functional recipe.
Tree feature indices address membership predicates, rather than raw category
codes. Threshold 0.5 sends nonmembers left and members right. Leaves evaluate
original input columns. An explicit feature-population root is required; generated
raw values, frozen subsets and leaf parameters survive recipe JSON and archives.

`sample_category_subset_splits` chooses each feature uniformly from the supplied
vocabulary slots, then independently includes each of its codes with the declared
probability (default 0.5). This is an r0 extension, not a fitted split search or a
claimed paper reconstruction. Feature selection occurs immediately before that
split's membership draws. Empty vocabularies, empty subsets and full subsets are
valid; no repair or rejection is performed. Unseen codes take the nonmember path.
The owned NumPy stream rolls back if construction fails. Controls are bounded to
1024 feature slots, 1024 splits and 1024 distinct exactly representable integer
codes per vocabulary. A code need not occur in a particular generated table.

For example, a categorical column with codes 0..3 and Gaussian column `z` selects
`-3*z + 0.5` for codes {0, 2}, and `2*z - 1` otherwise. This demonstrates a
noncontiguous categorical decision combined with model leaves; the codes' ordinal
magnitudes do not determine the branch. Tests compare saved generated outputs
against this independent arithmetic and reopen the recipe and dataset archive.
The sampler's feature and membership draws have separate numeric oracle checks
at inclusion probabilities 0, 0.3 and 1.

The compiler accepts authored recursive topology and models. The split sampler
returns frozen predicates, not a complete task-construction receipt. Automated
sampling of the whole categorical topology/model portfolio and its per-task CLI
construction transport remain open. The existing typed observation delivery work
in #30 is also separate: raw categorical population values here do not claim that
all downstream learner schema/encoding workflows are complete.

```sh
r0 generate --config configs/generation/categorical-model-tree-cpu.json --out /tmp/categorical-model-trees
```
