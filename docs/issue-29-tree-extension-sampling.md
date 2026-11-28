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
  depth zero is a single model leaf. Authored unbalanced trees remain separate.
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
sampler. Save `asdict(sampled)` as a construction receipt alongside those archives
when the request and construction RNG provenance are needed; the current generic
archive does not automatically embed this extra receipt. JSON receipt decoding
checks metadata, declared caller limits, frozen recipe and construction replay.
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
whole extension portfolio. Sampled categorical subsets, unbalanced extension
profiles, model-tree ensembles, generic archive receipt integration and broader
joint diversity/resource coverage remain open under #29. Full current-head
verification and the remaining #28/#30 work are separate gates.
