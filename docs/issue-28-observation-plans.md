# Feature-node observation planning candidate

`FeatureGroup` records an ordered named group and its feature count.
`sample_joint_observations` creates a `JointObservationPlan` for either all graph
nodes or separately sampled eligible-node subsets per group. In subset mode,
`n_feature_nodes` is a named integer decision with bounds 1 through the total
node count, using the shared context's local/meta/global policy. The all-node
mode consumes no named count decisions.

`materialize_joint_observations` retains the complete raw node permutation for
each subset, then draws feature assignments with replacement from the selected
prefix. All-node mode uses natural node order and draws assignments directly.
Group order and feature order are explicit and stable. Repeated node assignments
are valid. No edge, ancestry, target-connectivity, or predictability filter is
applied. This preserves valid disconnected cases required by #28.

The realization binds raw laws, shapes, ownership, sequence and stream chain,
and derives the assignments from those records. Replay requires an injected
matching pre-state and verifies every draw before publishing caller state.
Failures restore the caller stream. Named decisions and raw choices remain
separate evidence; structural validation alone does not prove RNG provenance.

## Source and limits of delivery

The inspected cached TabICLv2 `_dataset.py` selects either all nodes or a random
per-group subset and assigns features with replacement. `_base.py` defines the
named `n_feature_nodes` count law. The original source traverses a set of group
names; r0 uses an explicit ordered tuple for reproducibility. This is a documented
ordering choice, not a claim of bitstream identity with that source.

Cached source: `/tmp/node-pipeline-sampling-research/_dataset.py`, SHA256
`7e4237e2f8c34b98000c147b81383b33711b381073d78212191d0761685ec1e1`.
No upstream code was copied. Raw uniform index selection uses the existing
integer ledger law; its indices select from the retained eligible-node list.

This is a component for I2.graph.05. Converter selection and channel allocation,
native categorical features, hidden-channel uncertainty experiments, integrated
archive provenance and local/Colab controls remain unfinished. Feature counts
here do not stand in for converter input widths or node latent widths.

## Verification

Metadata admission bounds nodes, groups, features and complete permutation
storage before materialization. Raw ledger limits also apply. These estimates
are not measured peak memory.

Reproduction in `/tmp/r0-28-observation-plans`:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_joint_observation_plans.py \
  tests/priors/test_joint_width_plans.py
```

`/tmp/r0-28-observation-plans-red.log` retains missing-API failures. The first
implementation run rejected canonical integer records because its dtype check
used `int64` instead of the ledger's `<i8`; that failure is retained in
`/tmp/r0-28-observation-plans-green.log`. The repaired five-engine/all-node/subset
checks, replay-state tampering, raw budget rollback, and width checks passed:
15 cases in `/tmp/r0-28-observation-plans-engines.log`.

Independent review, full-suite checking of this component, integrated generation,
scientific distribution checks, browser/Colab work and portable evidence publication
remain pending. No issue or option is closed by this component-level result.
