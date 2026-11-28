# Exact category-subset tree predicates

`category_subset_activation((0, 2))` returns a frozen membership predicate: exactly
0 or 2 maps to 1; all other finite values map to 0. This is an r0 extension, not
an ordinal threshold or a learned category encoding. Empty subsets are valid and
produce a constant false predicate. Unseen codes and nonintegral inputs are
nonmembers; no category map is fitted from evaluation rows.

Codes must be distinct integers, with absolute value at most 2**53, and a subset
may contain 0–1024 codes. Evaluation rejects codes that the input tensor dtype
cannot represent exactly. NaN/infinity follow the existing explicit activation
failure rule. The implementation loops over codes with one Boolean result buffer
rather than allocating rows-by-columns-by-categories intermediates.

The predicate is an ordinary versioned `CausalActivation` record. To construct a
categorical subset split, select the raw feature with a linear projection, apply
the predicate in a node pipeline and split the resulting bit at 0.5 with the
existing frozen tree. Members route right, nonmembers left; exchanging leaf
values exchanges the outcome. Different subset predicates may feed distinct tree
split directions. Retain original categorical features as observed inputs and
keep the predicate bits and tree mechanisms privileged.

The saved example has raw category codes 0,1,2,3 with probabilities 0.1,0.2,0.3,0.4,
a Gaussian nuisance feature and subset {0,2}. Members receive target 3 and
nonmembers -2. There is no task rejection to remove a rare or absent category.
The example uses the existing source-root law and functional recipe formats:

```sh
r0 generate --config configs/generation/category-subset-tree-cpu.json --out /tmp/category-tree-datasets
```

The local CPU run saved both requested 64-row tasks. Focused checks independently
compare membership to NumPy, cover empty/noncontiguous/negative/large subsets,
unseen and noninteger values, reject duplicates and precision loss, reconstruct
actual generated labels and verify JSON/ZIP replay and repeated-row traces.
This delivers an authored categorical split component and a usable dataset
configuration. Automatic subset sampling and the full extension portfolio remain
open. Broader typed observation/schema/encoding delivery belongs to #30; this
component does not claim to finish it.
