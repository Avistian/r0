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
| T29-06 Extensions | Authored soft trees now compile sparse oblique splits and constant/linear/activated vector leaves into existing frozen networks. See the soft-tree contract. Exact hard oblique routing with either equality rule now uses projection plus frozen recursive traversal. Exact authored categorical-subset predicates and a generated tree example now exist. Hard-routed model leaves now compile constant/linear/activated scalar or score-vector models with exact routing. A bounded balanced hard/soft sampler now retains request, snapshots and compiled recipe with JSON replay. Sampled hard/soft pruned shapes now add an explicit depth-preserving unbalanced law. Sampled categorical portfolios, ensembles of model trees and broader combinations remain open. Per-task construction now embeds its receipt in dataset archives and checks request-bound resume identity. |
| T29-07 Populations/hyperpriors | Hard oblique and model-leaf helpers now retain existing joint, mixed/source, sampled-source and prototype roots; five population cases verify matched rows and archives. Broader hyperprior/dependency coverage and diversity audit remain open. |
| T29-08 Extremes/failures | Existing narrow tests cover some edge cases. A saved standalone inventory with independent expected paths/values and diversity/resource reporting remains open. |

[Soft-tree contract and current limits](issue-29-soft-trees.md).
The authoritative full requirements are [#29](https://github.com/Avistian/r0/issues/29).
Do not infer that an unlisted paper option is complete or deferred.

[Hard oblique task contract and exact equality rules](issue-29-hard-oblique-trees.md).

[Exact category-subset predicates and generated tree example](issue-29-category-subsets.md).

[Hard model-leaf task contract](issue-29-model-leaves.md).

[Tree feature-population integration](issue-29-tree-populations.md).

[Sampled hard/soft extension portfolio and remaining limits](issue-29-tree-extension-sampling.md).
