# Fitted structural trees: source laws and integration proposal

Research date: 2026-11-27. Documentation only; **fitted trees are not claimed
implemented or verified**. Authority is [accepted #28 spec](../specs/0028-causal-network-generators.md),
particularly decisions 12, 17–19, and [option inventory](0028-source-option-inventory.md)
(record group `0028-source-options-05.json`, fitted DT/XGBoost/RF/ExtraTrees/GBDT).
Standalone generator breadth remains #29; fitted structural components remain #28.

## TabICL paper and pinned implementation are distinct contracts

[TabICL v1, Appendix B.2](https://arxiv.org/html/2502.05564v1#A2.SS2)
fits XGBoost multi-output regressors to layer inputs and independent standard-normal
vector targets, then predicts those same inputs. Its depth and estimator expressions
are capped at four, with offsets two and one respectively and exponential rate 0.5.
The paper does not explicitly state the integer conversion. These are regressors
inside a classification-data generator, not fitted classifiers. Replacing the fitting
batch by independent calibration is an r0 adaptation.

The following code findings refer exclusively to
[`soda-inria/tabicl@9530bd65a7b806f6ad4f9f4335af745577e9d3cc`, `_tree_scm.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py).
This later pin is not proof of the historical paper/checkpoint implementation.

| Component | Exact disclosed code behavior |
| --- | --- |
| Fit inputs | `TreeLayer.forward` uses incoming parent matrix, applies `nan_to_num(0.0)`, transfers to CPU, fits and predicts on that matrix. NumPy iid `N(0,1)` fake targets have shape `(rows,out_dim)` and are redrawn each forward. |
| DT | `MultiOutputRegressor(DecisionTreeRegressor(max_depth=D, splitter="random"), n_jobs=-1)`; independent estimator per target channel. |
| ExtraTrees | `MultiOutputRegressor(ExtraTreesRegressor(n_estimators=T,max_depth=D),n_jobs=-1)`. |
| RF | `MultiOutputRegressor(RandomForestRegressor(n_estimators=T,max_depth=D),n_jobs=-1)`. |
| XGBoost | `XGBRegressor(n_estimators=T,max_depth=D,tree_method="hist",multi_strategy="multi_output_tree",n_jobs=-1)`. Joint vector leaves, unlike the independent-output sklearn wrappers. |
| Samplers | Independently per layer: `D=min(4,2+floor(Exp(scale=1/tree_depth_lambda)))`; `T=min(4,1+floor(Exp(scale=1/tree_n_estimators_lambda)))`. Default rates 0.5. |
| Architecture override | Constructor unconditionally forces `is_causal=False`, `num_layers` uniform on `{1,2}`, hidden width uniform on `{3,...,9}`. It creates that many hidden tree stages and one additional output tree stage; root width equals observed feature count. |
| Noise | First tree has no noise wrapper. Later/output stages append Gaussian noise. Fixed `noise_std`, or channelwise `abs(N(0,noise_std))` drawn once as its scale when `pre_sample_noise_std=True`; default scale 0.01. |
| Returned observation | Noncausal path returns original causes and last layer output. Dormant causal-selection branches do not undo the constructor override. |
| Failure behavior | After forward, any NaN in features/target replaces all features with zero and all targets with -100. This is upstream behavior, not acceptable silent r0 failure handling. |

No explicit `random_state` is passed to these estimators. All unspecified fit
parameters inherit installed library defaults; their historical versions/defaults
are **UNKNOWN** here. Do not interpret `n_jobs=-1` as an r0 resource authorization.
For example, the local RF defaults use bootstrap, whereas local ExtraTrees defaults
do not; the parent source only names the estimators.

[`_prior_config.py` at the same pin](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_prior_config.py)
selects XGBoost and both rates 0.5. Root sampling is a `meta_choice` over normal,
mixed and uniform, with separate choices for pre-sampled cause/noise statistics.
Noise scale uses `meta_trunc_norm_log_scaled`, mean bounds 0.0001–0.3, lower bound
zero and no rounding. These are sampler names/parameters, not permission to flatten
them into uniform distributions. Tree roots are supplied by `XSampler`; use the
existing root-law inventory for its complete mixed-marginal contract.

[`_activations.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_activations.py)
freezes first-batch means and sample standard deviations plus `1e-6` in
`StdScaleLayer`; that lifecycle does **not** freeze `TreeLayer`, which fits every
call. [`_mlp_scm.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_mlp_scm.py)
uses random neural parameters and activation modules. Neither file supplies a
fitted-classifier tree implementation or GBDT variant. Tree source comments about
activation/linear layers do not override its executable tree-and-noise sequence.

## Mitra v1: separate fit population and native classifiers

[Mitra v1 A.2.1–A.2.2, Algorithms 2–5](https://arxiv.org/html/2510.21204v1#A1.SS2):
DT/ET/RF/GB use sklearn classifiers/regressors. Algorithm 2 fits once on B synthetic
rows, predicts N new feature rows, returns predictions, not probabilities. Gaussian
continuous features; categorical count `floor(p_cat*(d+1))`, uniform index subset;
category cardinalities geometric, values multinomial. Regression targets normal;
classification targets use the categorical process, without quantile bucketing.
B, estimator settings/samplers, Gaussian parameters, geometric parameter,
multinomial probabilities and full `p_cat` range are **UNKNOWN** in this disclosure.

Algorithms 3–5 describe a separate **direct** family: uniform split feature,
uniform threshold within tracked ancestor interval; left branch `<=`; mandatory
minimum depth, then stopping probability until maximum depth; normal regression
leaves; classification leaves cycle modulo class count. Prose randomizes the first
class; Algorithm 4's zero-initialized counter ignores its sampled start. Prose
samples a fixed function; Algorithm 5 samples trees inside the row loop. Preserve
named prose/algorithm variants. Forest outputs use mean/regression or majority
vote/classification; tie law is **UNKNOWN**. These direct algorithms cannot satisfy
the fitted-family requirement.

## Newer hybrid papers: disclosed structure, incomplete fitting recipe

[Mitra-v2 v1 §2.3](https://arxiv.org/html/2609.04540v1#S2.SS3)
includes DT/ExtraTrees/RF among hybrid mechanisms. It concatenates vector-valued
parents before applying a node mechanism. Multiple-parent nodes also project each
parent to output width and aggregate these residuals with the mechanism result
(mean, softmax-weighted sum, MLP, product, maximum). It standardizes/scales/normalizes
nodes and quantile-buckets the selected target for classification. The section does
not disclose fit-table size, fake-target law, tree depths/counts, estimator classes
or fit/refit lifecycle: **UNKNOWN**, not implicitly inherited from TabICL or Mitra-v1.
Outer-mixture gradient boosting (§2.2) does not establish a hybrid-node GBDT recipe.

[O’Prior v2 §2.1](https://arxiv.org/html/2605.18971v2#S2.SS1)
describes tree regressors fitted at each layer, extending XGBoost with DT,
ExtraTrees, RF and a direct variant. It does not disclose exact fake-target law,
fit parameters or library versions: **UNKNOWN**. Its overall contract requires
fitted statistics to depend only on support rows. This does not establish an
independent-calibration implementation or repeated-call freeze semantics. Keep
support-only fitting, independent-calibration fitting and batch-refit references
separate. Neither newer paper discloses a fitted XGBoost classifier structural
mechanism in the inspected sections; such an option would be an r0 extension.

## Locally available fitting backends and output semantics

Read-only inspection on 2026-11-27 used
`/tmp/r0-28-pipeline-integration/.venv/bin/python`: sklearn **1.9.1** imports;
DT/ET/RF/GradientBoosting classifier and regressor classes all import. XGBoost's
module spec is absent. `pyproject.toml` declares `scikit-learn>=1.9.1`, does not
list XGBoost, and forbids priors importing LightGBM. No installation, fitting,
implementation, dependency edit, cloud call or dataset download was performed.

A concrete exporter must preserve estimator prediction semantics, not merely tree
shapes. sklearn's [RF classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
averages per-tree class probabilities, then predicts the class with greatest
probability. Majority vote over tree argmax labels is a different function.
Retain class ordering and normalized leaf class masses; absent classes and ties
must remain observable. [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
uses additive stage scores with a loss-specific link and initial estimator;
multiclass stages contain one tree per class, binary stages one tree. A mean of
its tree predictions is wrong. Regression boosting likewise needs the initial
prediction, learning rate and additive trees. Record full explicit parameters
rather than relying on future installed defaults.

[XGBoost vector-leaf documentation](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)
confirms `multi_output_tree` with `hist` creates vector leaves, whereas default
`one_output_per_tree` fits separate output models. A sklearn GBDT replacement
cannot be reported as that XGBoost recipe. XGBoost fitting/export remains
**PENDING** under the current dependency constraint; either an explicitly named
clean-room mathematical adaptation or a separately authorized dependency decision
is needed. The former would require its own numerical comparison evidence, not
merely a compatible family name.

## Proposed immutable split-table integration (r0 adaptation, not delivered)

1. **Fit boundary.** A plan declares family, regression/native classification,
   output width/classes, explicit estimator parameters, input law, independent
   base-table size and named RNG streams. Generate a separate base population
   through already frozen ancestor mechanisms in topological order. Never use
   episode/query rows or their labels as fake-fit targets or calibration inputs.
   Fit once on CPU with `n_jobs=1` and explicit per-output/per-estimator seeds.
   Keep fit-target RNG separate from structural-noise and episode RNG streams.
   For TabICL adaptation use independent standard-normal target channels; for a
   Mitra-inspired bounded reconstruction explicitly choose/persist otherwise
   undisclosed Gaussian/geometric/multinomial parameters.
2. **Frozen data.** Export owned immutable arrays/tuples for feature indices,
   thresholds, left/right children, leaf values/class masses, ordered output/class
   maps and ensemble combiners. Keep estimator/version, full parameters, source
   pin, adaptation ID, fit-input/target hashes, seeds, sizes and schema hash.
   No pickle or live estimator belongs in an execution/archive payload. Export
   regression means or independent-channel trees, classifier probability
   averaging, or boosted initial-score-plus-weighted-sum/link as distinct schemas.
   Shared vector split topology must be representable for eventual XGBoost.
3. **Traversal semantics.** Persist `<=` versus `<`, comparison/input casting,
   threshold dtype, missing-value routes and leaf-value dtype. sklearn conversion
   of inputs to float32 can change boundary routing: checking float64 input
   against exported float64 thresholds alone is not equivalent. A missing-route
   capability must be implemented explicitly or rejected by schema validation;
   do not silently substitute zero. Preserve r0's numerical-failure accounting.
4. **Proposed exploratory bounds.** Initial named bounded recipe: base rows
   32–1024; parent width 1–256; output width 1–16; depth 1–8; ensemble stages 1–32;
   classifier cardinality 2–100, subject to total-tree/node and fit-cell caps.
   Suggested global per-mechanism limits: 4096 trees, 131072 nodes and 1048576
   fit-input/target cells. Validate products *before* allocation, including
   output/class multipliers and full binary-tree upper bound `2^(depth+1)-1`.
   These are proposed engineering limits, **NOT_MEASURED**, not source ranges or
   throughput claims. Preserve the TabICL 1–4/2–4 sampler as a separately named
   adaptation. Larger recipes need resource evidence; the maxima are not a
   promised Cartesian product. Reject explicit budget excess; do not redraw
   valid difficult fits until an easier sample appears.
5. **Replay and interventions.** Persist frozen split tables and combiners so
   replay does not rerun sklearn, depend on sklearn availability or resample
   targets. Parent interventions traverse the same table; only an explicit
   mechanism edit may refit. Expose fitting as a generation-process operation,
   separately from causal edges and tree split paths. UI/archive shows source
   versus adaptation, base population, frozen identity, fit cost, and failures.
6. **Required independent evidence.** Compare exported evaluator with backend on
   retained synthetic probes, exact thresholds, adjacent representable values,
   out-of-range inputs, class ties/absent classes and multiclass boosting. Check
   chunking/permutation/query append invariance, save/reopen without backend,
   fixed-table interventions and CPU/GPU routing. Retain a valid constant/weak
   fit; disclose single-class boosted-classifier failure without silently adding
   labels or resampling. Measure fit/export bytes/time before widening bounds.
   These checks, browser/Colab delivery and family coverage are **NOT_RUN** here.

A separately named batch-reference mode may reproduce TabICL's refit behavior,
but must retain each batch's fitted state and declare loss of cross-batch function
identity. It cannot be the training-ready default. The preceding proposal offers
a common structural integration surface for #28 and future #29 reuse; it does
not resolve undisclosed source settings or establish full-prior reproduction.
