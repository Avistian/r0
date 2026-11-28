# Issue 29 generator audit in progress

The full issue remains open. This initial map identifies reusable components and
known gaps; it is not family-level acceptance. Generator delivery now includes
#28, #29 and #30; presentation is deferred to #126.

| Requirement | Inspected implementation and remaining work |
| --- | --- |
| T29-01 Direct trees | `RecursiveForestPlan`, sampled recursive forests and retained path intervals exist. Still audit standalone API/CLI paths, split-law alternatives, explicit equality variants, all budgets and numeric examples. |
| T29-02 Oblivious trees | `JointTreeFunctionRealization` retains selected features/threshold rows and Gaussian vector leaves. Still audit standalone construction, preference laws and matched-budget comparisons. |
| T29-03 Fitted trees | DT/ET/RF/GB and XGBoost fit/export primitives exist, including parameter samplers and independent/population lifecycles. Still audit standalone profiles and source-specific fit distributions. |
| T29-04 Outputs | Scalar/vector and native-classifier primitives exist. Probability-leaf sampling, frozen score adapters, noise and absent-class coverage need requirement-by-requirement audit. |
| T29-05 Ensembles | Recursive mean/vote and fitted probability/boosting evaluators exist. Correlated forests, weighted reductions and standalone disagreement/correlation evidence remain to audit. |
| T29-06 Extensions | Authored soft trees now compile sparse oblique splits and constant/linear/activated vector leaves into existing frozen networks. See the soft-tree contract. Exact hard oblique routing with either equality rule now uses projection plus frozen recursive traversal. Categorical subsets, hard model leaves, sampled portfolios and combinations remain open. |
| T29-07 Populations/hyperpriors | Existing root and scalar-distribution primitives are candidates for reuse. No full standalone tree integration audit yet. |
| T29-08 Extremes/failures | Existing narrow tests cover some edge cases. A saved standalone inventory with independent expected paths/values and diversity/resource reporting remains open. |

[Soft-tree contract and current limits](issue-29-soft-trees.md).
The authoritative full requirements are [#29](https://github.com/Avistian/r0/issues/29).
Do not infer that an unlisted paper option is complete or deferred.

[Hard oblique task contract and exact equality rules](issue-29-hard-oblique-trees.md).
