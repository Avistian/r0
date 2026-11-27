# Independently fitted structural trees

This slice implements actual synthetic fitting for DecisionTree, ExtraTrees,
RandomForest and GradientBoosting regression and classification. It adds a
separate frozen split-table function; the existing directly sampled oblivious
tree family is unchanged. The source contracts and disagreements remain in
[the source-law note](research/0028-fitted-tree-source-laws.md).
This is a bounded r0 adaptation, not a reproduction of any complete paper prior.

## Numerical contract

`FittedTreePlan` selects `dt`, `extra_trees`, `rf` or `gbdt`, a regression or
classification task, geometry, fit rows, depth and estimator count. Each output
is fitted independently. Regression targets are iid standard normal;
classification targets are iid uniform integer class codes, not quantile bins.
Standalone inputs are iid standard normal. Graph fitting instead uses an
independent population drawn through already frozen ancestors. No episode,
query, future label or intervention population is a fitting input.

The DT splitter is explicitly `random` by default and can be `best`. RF uses
bootstrap; ExtraTrees does not. Both use one CPU job. Classification uses sqrt
feature selection; regression uses all features. All actual sklearn parameters,
including library defaults, are serialized with the sklearn version and each
output's explicit seed. These local parameters are an r0 recipe, not inferred
historical defaults. sklearn's internal random draws are **not** claimed captured;
the seed and algorithm version are their reproducibility metadata.

Every fitted estimator is exported to immutable tuples of feature indices,
float64 thresholds, ordered child indices and leaf values. Rootedness,
reachability, depth, dimensions and class normalization are checked. Finite
inputs convert to float32 before `<=` comparison against float64 thresholds;
overflow rejects rather than silently switching arithmetic. Missing/nonfinite
input routing is rejected. Replay uses PyTorch split traversal and never fits,
unpickles, or imports sklearn through the fitting backend.

Regression forests average predictions. Classifier forests average leaf class
probabilities, then argmax in retained class order. `classification_rule="majority"`
is a separately named extension counting per-tree argmax votes; it is not
sklearn's classifier rule. Ties choose the first retained class. Absent target
classes stay absent in the class map; single observed classes are valid for
nonboosted classifiers. GBDT retains actual constant initializer raw scores,
ordered stages, learning rate and identity/sigmoid/softmax link. It does not
average trees. A failed boosted fit, including a single-class target population,
is surfaced without relabeling or resampling.

`evaluate_frozen_function` produces one prediction/code per output channel.
`inspect_fitted_tree_scores` exposes regression predictions or class scores in
each output's retained order: probabilities for ordinary classifiers and vote
fractions for the named majority variant. Native classifier structural functions
are not themselves a claim that the enclosing task uses a native categorical
outcome: an ordinary `FunctionalCausalRecipe` still uses its declared task adapter.

## Sampling, archives and inspection

```python
import numpy as np
from r0.modules import priors

plan = priors.FittedTreePlan(
    family="rf", input_width=3, output_width=2, rows=128, depth=3, estimators=4
)
fit = priors.fit_fitted_tree(
    plan,
    input_rng=np.random.default_rng(41),
    target_rng=np.random.default_rng(42),
    backend_rng=np.random.default_rng(43),
)
# fit.function is directly usable as a FunctionalCausalNode.function.
```

For a frozen skeleton, `sample_fitted_graph(skeleton, ((node_index, plan), ...),
prefix_rng=..., input_rng=..., target_rng=..., backend_rng=...)` replaces selected
nonroot functions in topological order. A sampled graph's `.runtime` can serve as
the explicit skeleton. Each downstream fit sees the newly frozen upstream fitted
function. The sampled selector/topology construction archive stays separate;
this recipe does not pretend to preserve its omitted original selector history.

The resulting `FittedFunctionalCausalRecipe` goes through `generate_task`,
`recipe_from_dict`, the ordinary recipe JSON archive and `explain_generation`.
Its mechanism card identifies independent fitted construction. Inspection shows
the full split tables and combiners as function parameters; retained fitting
populations and raw draws are construction metadata in the recipe, not causal
edges. The fit record binds each function to its preceding ancestor fingerprint.
Reopening verifies ancestor populations with the retained prefix stream and
verifies input/target/seed raw draws and final RNG states, without refitting.
Frozen-table inference remains available separately without the fit backend.

Every construction RNG is injected. Input, target, backend-seed and ancestor
streams must be distinct. Failed fitting or final graph validation leaves caller
streams unchanged. Raw fitting draws retain values, laws, parameters, ownership,
engine and before/after state digests. Snapshot records retain NumPy version.
CPU fitting and tensor inference are distinct operations. Queries, chunking and
row permutations cannot change the immutable fitted function.

Selected replacement nodes currently cannot carry pre-fitted pipelines, and the
skeleton is an uncalibrated functional recipe. This prevents stale normalization
from being silently attached to a new fitted equation. A wider pipeline/native
construction surface remains required integration work.

## Bounds and evidence

The explicit exploration envelope is 32–1024 synthetic fitting rows, 1–256 input
channels, 1–16 independent output channels, depth 1–8, 1–32 estimators and 2–100
requested classes. Pre-fit admission checks worst-case tree/node products:
4096 trees, 131072 nodes and 1048576 input/target cells. Caller `CausalLimits`
can tighten these and charge fit tables, raw metadata and retained output tables.
These bounds are not a promised Cartesian product or measured operating envelope.

Public tests independently refit sklearn from retained data/parameters and compare
regression outputs and classifier probabilities, including exact thresholds,
adjacent float64 values and float32 neighbors. They cover all four families,
binary/multiclass, probability-vote disagreement, absent classes/ties, query
append/permutation stability, graph generation/inspection and disk JSON reopen,
raw metadata tampering, five NumPy engines and failed-fit RNG rollback.

A local 12-cell CPU measurement uses 128 rows, 3 inputs, 2 independent outputs,
depth 3 and 4 estimators (DT has one tree per output). The retained reproduction
script and measurements are ignored files under
`docs/evidence/issue-28/fitted-trees/`. They report elapsed fit/export time,
serialized record bytes and process peak RSS including imports and earlier cells.
These are single-process observations, not bulk throughput or per-fit isolated
memory measurements. Broader bounds, GPU, browser and Colab evidence remain
**NOT_RUN** for this slice. Full integration verification is owned by the lead.

## Required options still open

| Requirement | Status and distinction |
| --- | --- |
| XGBoost `hist` joint vector-leaf structural fit | **PENDING**. Package absent; no new dependency authorized. sklearn GBDT is not this algorithm. |
| Faithful XGBoost path | Once dependency authority exists, fit the pinned CPU `hist`/`multi_output_tree` model, export shared vector leaves, base score and additive objective/link; compare JSON exporter against backend on boundaries, missing routes and vectors. No such backend comparison has run. |
| Clean-room histogram boosting alternative | Would be a separately named mathematical adaptation needing histogram/gradient/Hessian/split-gain and vector-leaf oracle evidence. Not implemented or claimed equivalent. |
| TabICL capped exponential depth/count law and graph overrides | Exact reference remains inventoried; this plan uses explicit caller settings, not those joint samplers. |
| TabICL batch refit on actual parent rows | Reference mode remains required; default here intentionally fits independent rows once. No full TabICL reproduction claim. |
| O'Prior support-only fitting | Distinct lifecycle, still required; independent rows are not support rows. |
| Mitra mixed categorical/Gaussian synthetic fit population | Still required; this slice's standalone input law is Gaussian. Published missing parameters remain UNKNOWN. |
| Mitra direct tree stopping/interval and fresh-per-row variants | Required separate direct families; fitted sklearn trees do not satisfy them. |
| Source joint selector and parent-composition integration | New frozen family registry is ready; the older eight-family named joint selector is unchanged. |
| Native categorical-node target/pipeline transport | Structural classifier outputs are available; complete typed sampled/native and exploration delivery remains required. |
| Wider resource and scientific claims | PENDING measurements; no predictive superiority or full-prior fidelity claim. |
