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

The compiler accepts authored recursive topology and models. The standalone split
sampler returns frozen predicates. Complete sampling and per-task construction
receipts are now available through the tree extension profiles described below.
The existing typed observation delivery work
in #30 is also separate: raw categorical population values here do not claim that
all downstream learner schema/encoding workflows are complete.

```sh
r0 generate --config configs/generation/categorical-model-tree-cpu.json --out /tmp/categorical-model-trees
```


## Complete sampled categorical profiles

`TreeExtensionRequest.families` now accepts `categorical` and
`categorical-pruned`, alongside the four continuous-split families. These require
an uncalibrated, ordinary `SourceRootBase` containing at least one categorical
coordinate. Gaussian/other coordinates remain available to model leaves, but are
excluded from split-feature selection. Category cardinalities and frequencies
come from the declared root law. Calibrated, prototype and sampled-source root
variants are not admitted for these two profiles.

Choose the family and depth first, then the balanced/pruned shape using the
existing topology laws. For each internal node, choose uniformly among eligible
categorical columns and include each code independently with probability
`split_density`. Preserve empty/full subsets. Membership thresholds are exactly
0.5 in [0, 1]; no Gaussian split coefficients, random thresholds or temperatures
are drawn for categorical families. Those continuous-split controls are inactive
for these families. Leaf type/coefficient/bias laws remain unchanged. Leaf models
still use all original columns, including raw category codes. The equality setting
cannot affect 0/1 membership compared with 0.5. A depth-zero task is a single model
leaf and consumes no split draws.

The complete request, compiled predicates/models, and initial/final generator
states are retained by `SampledTreeExtension`. Existing `tree-extension` run
construction produces fresh per-task trees and embeds the receipt in the archive.

```sh
r0 generate --config configs/generation/sampled-categorical-tree-cpu.json --out /tmp/sampled-categorical-trees
```

Both profiles are checked at depths zero, one and three. A separate draw oracle
checks eligible-feature mapping and membership decisions. Guarded generation,
archive receipt reopening and zero-generation resume are checked together. This
adds a named r0 categorical portfolio; mixed categorical/oblique splits within
one tree, further topology priors and automatic ensembles remain open.
