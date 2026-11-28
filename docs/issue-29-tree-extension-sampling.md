# Sampled hard/soft model-tree portfolio

`TreeExtensionRequest` and `sample_tree_extension(request, rng=...)` provide a
bounded r0 extension sampler over the implemented hard/soft tree compilers.
This is not a named paper prior. The returned `SampledTreeExtension` contains the
request, realized family/depth, compiled recipe, NumPy initial/final snapshots
and resource limits. `validate_replay()` reconstructs the complete construction;
`sampled_tree_extension_from_dict` reopens and validates its JSON receipt.

The explicit sampling laws are:

- Family is uniform over `families` slots; defaults are hard and soft. Repeated
  slots give deliberate categorical weights.
- Depth is uniform over the inclusive integer range `min_depth..max_depth`,
  bounded by 0..4. The sampled topology is a complete balanced binary tree;
  depth zero is a single model leaf. The optional pruned profiles below sample
  unbalanced shapes.
- Split coefficients are independent zero-mean Gaussians with standard deviation
  `coefficient_scale`, multiplied by independent Bernoulli `split_density` masks.
  All-zero directions are valid and are not repaired or redrawn.
- Thresholds are uniform on ±`threshold_bound`. Soft temperatures are log-uniform
  within `temperature_bounds`. Temperatures are drawn even for hard realizations,
  keeping the declared construction sequence explicit.
- Leaf types are uniform over `leaf_families` slots: constant, linear, tanh or sine
  by default. Coefficients and biases are independent zero-mean Gaussians with
  the configured scale. Constant leaves retain only the biases; coefficient draws
  are still consumed. Each leaf has fresh parameters.

Hard recipes preserve the requested equality rule and use frozen exact traversal;
soft recipes use sigmoid routing and average at equality. Existing joint/source/
prototype populations can be supplied through `root`. All draws use the caller's
owned stream, and construction failure rolls it back. Worst-case geometry and
metadata limits are admitted before drawing. No valid weak or constant task is
silently rejected.

The `.recipe` is immediately usable with `generate_task` and the CLI. Dataset
archives retain its realized function and replay values independently of the
sampler. The archive also retains the construction receipt when using
`GraphConstruction` topology
`tree-extension` with a complete serialized `TreeExtensionRequest` under
`request`. This path draws a fresh tree for each episode seed/task ID from a
domain-separated PCG64 stream that also includes the construction seed. Resume
checks bind the receipt and compiled recipe to those controls and exact integer
identities. Direct generation of a fixed `.recipe` still needs a separate receipt
(`asdict(sampled)`). JSON receipt decoding checks metadata, declared caller limits, frozen recipe and construction replay.
It requires the recorded NumPy environment rather than claiming cross-version
bit identity.

```sh
r0 generate --config configs/generation/sampled-tree-extension-cpu.json --out /tmp/sampled-tree-datasets
```

The example recipe was sampled with seed 29 and depth one. Its construction
receipt was saved and reopened before the local CLI generated both requested
64-row datasets. Seventeen checks cover five engines and both families, independent
split/threshold/temperature draw oracles, stumps/deeper trees, task archives,
construction JSON, tampering, preflight rejection and late-failure RNG rollback.
The ignored evidence packet retains initial type/regex and JSON-validator failures
as well as their corrected results.

This closes the initial balanced hard/soft model-tree sampling component, not the
whole extension portfolio. Further topology laws, automatic ensemble sampling and broader
joint diversity/resource coverage remain open under #29. Full current-head
verification and the remaining #28/#30 work are separate gates.


For fresh per-task trees with archived construction receipts:

```sh
r0 generate --config configs/generation/per-task-tree-extension-cpu.json --out /tmp/per-task-trees
```

This configuration generates three 64-row datasets, with depth sampled from one
through two and both hard and soft families eligible. Reusing the output directory
checks the retained construction identities before resuming. The CLI example
completed three tasks locally; focused archive tests check receipt reopening,
altered seeds, integer identity aliases, different task IDs and zero-work resume.


### Pruned shape profiles (r0 extension)

`families` may also contain `hard-pruned` and `soft-pruned`, including alongside
balanced entries. After family/depth selection, keep the all-left path to the
sampled depth; each other node above that depth splits with probability 0.5,
independently in breadth-first order. Stop at the depth bound. This guarantees
exact maximum depth, permits shallow off-spine leaves and occasionally produces
a fully balanced tree. The shape law intentionally favors the left spine; it is
not uniform over tree shapes and is not a claimed paper reconstruction. There
is no rejection or balancing pass. Depth zero and one consume no shape coins.

After shape selection, sample split and leaf parameters with the same declared
laws as balanced profiles. Internal nodes precede leaves in breadth-first order;
this ordering defines parameter allocation. Resource admission uses the complete
balanced tree as its worst-case bound. Existing `hard`/`soft` receipts retain
identical shape, draw order and compiled recipes.

```sh
r0 generate --config configs/generation/pruned-tree-extension-cpu.json --out /tmp/pruned-trees
```

The example selects hard/soft pruned trees at depth three. Shape checks retain an
independent two-coin numeric example and verify balanced layout/RNG preservation
at all supported depths. Both new families have receipt replay and dataset archive
checks at depths zero and three. These checks do not close categorical sampling,
model-tree ensembles or the whole #29 diversity inventory.

Complete categorical profiles and their distinct split laws are documented in
[Categorical model trees](issue-29-categorical-model-trees.md).
