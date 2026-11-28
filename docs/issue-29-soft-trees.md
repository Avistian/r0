# Soft trees with oblique splits and model leaves

This is an explicit r0 extension for T29-06, not a named paper prior.
`SoftTreeBranch(weights, threshold, left, right, temperature)` defines a split.
`compile_soft_tree(input_width, nodes)` accepts a rooted tree in which children
follow parents, and each leaf is a `NetworkAffine`. Sparse weights give sparse
oblique splits; a single nonzero weight gives an axis-aligned split. Leaf maps
can be constant, linear or use a declared activation such as tanh, sine or ReLU.
Vector leaves must agree in width. Trees may be balanced or unbalanced.

For input x, a branch computes z=(weights*x-threshold)/temperature and returns
`(1-sigmoid(z))*left(x) + sigmoid(z)*right(x)`. Descendant branch probabilities
therefore multiply along paths. Temperatures must be positive. A low temperature
approaches hard routing away from the boundary, but equality always averages the
two child outputs; it is not the hard tree's left-at-equality rule. With constant
leaves -2 and 3, threshold 0.25 and temperature 0.00001, probes 0.24, 0.25 and 0.26
produce -2, 0.5 and 3 to float64 precision.

The compiler emits the existing `FrozenNetworkFunction` with affine softmax gates
and leaf experts. `NetworkRecipe` makes it a standalone task; the function may
also be a structural graph mechanism. Existing JSON decoding, CLI recipe input,
dataset ZIP export/reload and numeric generation traces apply. Original authored
nodes are construction inputs; the archive retains their compiled coefficients,
connections and leaf models rather than introducing a second tree schema.

Compilation checks connectivity, ordered children, dimensions and conservative
parameter/channel/workspace limits before building dense gate/projection matrices.
No splitting or leaf sampling law is implied: callers supply the explicit tree.
The compiled representation is dense even when split weights are sparse, so sparse
inputs do not imply sparse memory use. All leaf models are evaluated; nonfinite
values are failures even for a leaf with a numerically tiny gate weight.

Independent NumPy calculations verify oblique gating, linear/tanh leaves and deep
vector-valued path products. Other checks cover temperature/boundary behavior,
invalid topology and limits, serialized equations and a generated standalone
archive with repeated-row numerical traces. This does not close T29-06: categorical
subset splits, sampled extension portfolios and full composition/diversity coverage
remain open. [Exact hard oblique routing](issue-29-hard-oblique-trees.md) now has
a separate implementation with explicit equality rules.

Run the saved two-feature example with:

```sh
r0 generate --config configs/generation/soft-oblique-tree-cpu.json --out /tmp/soft-tree-datasets
```

The example combines an oblique soft gate, a constant left leaf and a tanh model
right leaf. Its local CPU run generated and saved both requested 64-row tasks.
The retained ignored evidence includes CLI output and explicit numerical probes.
Five focused soft-tree tests plus existing network regressions passed (28 total);
whole-repository lint, formatting and type checks also passed. Full current-head
acceptance remains pending.
