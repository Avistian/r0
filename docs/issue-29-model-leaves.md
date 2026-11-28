# Hard-routed trees with model leaves

`model_leaf_tree_recipe(task, projection, tree, models)` creates an authored r0
extension with exact tree routing and one model per terminal node. Models are
`NetworkAffine` records in terminal-node record order, not traversal order.
Their input is the original feature vector; sparse oblique split directions are
specified separately by `projection`. Models may be constant, linear, or use
existing activations such as tanh, sine and ReLU. All leaf output widths must
match the task adapter: one scalar for regression/binary or a vector of class
scores for multiclass. This is not a native classifier-leaf probability API.

The original terminal values/labels are deliberately replaced by leaf IDs. The
existing exact hard-tree compiler selects an ID, frozen membership predicates
select its model, and existing product/sum graph operations assemble the output.
`equality="left"` and `equality="right"` retain the exact boundary rules documented
for [hard oblique trees](issue-29-hard-oblique-trees.md). At equality exactly one
model is selected; there is no soft interpolation.

All models are evaluated before masking. Nonfinite output from an inactive model
therefore fails explicitly; the compiler does not use lazy traversal to hide
invalid values. Dense compiled graph resource limits apply. This API constructs
one authored tree, not a sampled forest-of-model-trees prior. Automatic topology,
model-family and coefficient distributions remain part of the open extension
portfolio work. Existing constant-leaf forest reducers remain separate.

The resulting `FunctionalCausalRecipe` retains projection, routes, leaf-ID map,
models and selector operations through existing JSON/ZIP serialization. Raw
features are observed; route IDs and intermediate model values remain privileged.
The saved example uses direction x0-2*x1 at threshold 0.25. Its left model is
2*x0-1; its right model is tanh(3*x1+0.2).

```sh
r0 generate --config configs/generation/model-leaf-tree-cpu.json --out /tmp/model-tree-datasets
```

The local example generated and saved both 64-row tasks. Independent checks cover
literal model arithmetic, both equality rules, terminal-record mapping in a deeper
three-output tree, geometry/resource rejection, generated targets, JSON/ZIP replay
and repeated-row traces. Evidence under `docs/evidence/issue-29-model-leaves/`
retains an invalid two-class multiclass fixture failure before its correction to
three classes; no source behavior was changed to accept an invalid task.
Full current-head acceptance and the broader #29 portfolio remain open.

[Existing feature-population families](issue-29-tree-populations.md) can be passed
through `root`, with their frozen calibration and sampling provenance retained.
