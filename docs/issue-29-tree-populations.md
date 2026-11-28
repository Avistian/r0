# Feature-population variety in tree task helpers

Both `oblique_tree_recipe` and `model_leaf_tree_recipe` now accept the existing
`JointRootBase`, `SourceRootBase`, `SampledSourceRootBase` and `PrototypeRootBase`
records. The supplied root is retained unchanged in the compiled task; independent
root calibration, sampled profile choices and frozen prototype-bank provenance
are not replaced by a fresh Gaussian draw or discarded. The default remains the
existing Gaussian joint root.

This connects the helpers to mixed numeric/category populations, bounded
coordinates, Zipf tails, independent source profiles and clustered/shared-prototype
populations. Existing joint roots retain uniform/ball/Gaussian/covariance laws.
The helpers do not invent another feature sampler, query-derived normalization
or category encoding. All existing root limits and finite-moment requirements
continue to apply at task admission and evaluation. Observation effects and
learner-visible schema/encoding expansion remain #30 work.

Five population cases exercise both helpers: mixed uniform/categorical,
fully categorical, bounded plus unbounded Zipf, a three-point prototype bank,
and a sampled source profile. Each case verifies retained root identity,
JSON/ZIP replay, independent output arithmetic and repeated-row numeric traces.
Using the same root, seed and task ID produces exactly matched input rows between
the constant-leaf and model-leaf tasks. The prototype fixture also retains its
exact x1=2*x0 relationship. These are selected integration cases, not proof of
all possible distribution/model combinations or a statistical diversity report.

```sh
r0 generate --config configs/generation/mixed-population-model-tree-cpu.json --out /tmp/mixed-tree-datasets
```

This CPU example saved both requested 64-row datasets with a uniform first feature
and categorical second feature. Existing tree tests plus the population cases
passed (17 checks). The ignored evidence packet is
`docs/evidence/issue-29-tree-populations/`. Broader sampled portfolios,
hyperprior/dependency coverage and resource/diversity reports remain open.
