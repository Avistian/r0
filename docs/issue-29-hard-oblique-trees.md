# Exact hard oblique tree tasks

`oblique_tree_recipe(task, projection, forest, equality="left")` turns an authored
regression `RecursiveForest` into a standalone task with raw observed features,
linear split directions and the existing exact frozen tree traversal. Rows of
`projection` define directions; the forest's split feature indices select them.
Sparse rows provide sparse oblique splits, and one nonzero entry provides an
axis-aligned split. Multiple trees and repeated direction use retain the existing
forest's mean reduction and ancestor-consistent intervals.

`equality="left"` sends `dot(weights,x) <= threshold` left. `equality="right"`
sends only strict `<` left. The latter compiles by negating directions and
thresholds and exchanging children, including their interval bounds. No finite
softmax temperature or epsilon approximation is involved. All projections and
thresholds use the existing float64 numerical semantics; this is not a claim of
exact real arithmetic for arbitrarily ill-conditioned dot products.

The result uses ordinary `FunctionalCausalRecipe` records. Inputs expose the raw
root channels; projections and frozen mechanisms stay privileged. The compiled
projection, thresholds, routes, values and mean reduction survive JSON and dataset
ZIP replay. There is no new evaluator or serialization version. The optional
`root` accepts an existing joint root law; the default is Gaussian. Parameter and
channel/workspace admission precedes dense identity construction.

The forest must contain regression-vector values. Scalar outputs support regression
and the existing binary score adapter; matching multiclass vector outputs are
scores for the existing multiclass adapter. Native classifier forests are rejected
rather than silently treating their majority labels as score-based targets. This
is an authored r0 extension, not a sampled oblique-tree prior or a complete native
categorical-output workflow. Hard model leaves remain separate portfolio work.
[Category-subset predicates](issue-29-category-subsets.md) now provide an authored
component and a standalone generated example.

For the direction `x0 - 2*x1` and threshold 0.25 with constant leaves -2 and 3,
probes (0.24,0), (0.25,0), (0.26,0) give [-2,-2,3] with left equality and [-2,3,3]
with right equality. (2.25,1) is another boundary point. Six checks verify those
values, adjacent float64 values, repeated-direction interval transformations,
invalid geometry, actual generated labels and saved archive traces.

```sh
r0 generate --config configs/generation/hard-oblique-tree-cpu.json --out /tmp/hard-tree-datasets
```

The local CPU example saved both requested 64-row tasks successfully. Retained
ignored evidence lives under `docs/evidence/issue-29-oblique-tree/`. Full
current-head acceptance and the rest of #29 remain open.
