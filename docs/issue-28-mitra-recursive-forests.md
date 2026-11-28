# MITRA recursive interval forests — bounded core

The #28 M1.tree.09–19 family now has its own direct sampler. It is separate from
oblivious trees, sklearn fitting and XGBoost fitting. No training data or backend
is used. Graph adapters, guarded explorer/Colab delivery and hyperparameter prior
catalogs remain required follow-ups; this core does not close those obligations.

## Primary source and explicit reconstructions

[MITRA v1 A.2.2 and Algorithms 3–5](https://arxiv.org/html/2510.21204v1#A1.SS2.SSS2)
describe random split dimensions, ancestor-constrained uniform thresholds,
minimum/maximum depth and early termination, Gaussian regression leaves, cyclic
class labels, mean regression aggregation and majority classification aggregation.
Algorithm 3 sends equality left. The prose initializes class cycling randomly;
Algorithm 4 draws a start but assigns the counter modulo the class count. The
prose describes a frozen function; Algorithm 5 constructs forests per row.
These discrepancies remain selectable, not silently resolved.

The reconstruction uses path-local interval copies, depth-first left-first
visitation, a zero-based root depth, and a counter reset per tree. These resolve
unspecified ownership/counter details. Algorithm 4's split and threshold draws
are retained at leaves; the termination draw occurs whenever depth is below the
maximum, including below minimum depth. Threshold quantiles are NumPy `[0,1)`;
floating-point interval collapse is retained without retries. Gaussian scale
means standard deviation. Hyperparameter distributions and majority ties are
not fully disclosed: constructor defaults and tie choices are explicit r0
settings, not recovered source settings.

## Public controls and immutable records

`RecursiveForestPlan` exposes input width, estimator count, minimum/maximum depth,
terminal probability, explicit feature intervals, regression leaf means/scales,
class count/assignment, tie policy, source label, lifecycle and population count.
`mitra-prose-v1` selects random-start cyclic classes; `mitra-algorithm4-v1` selects
zero-start classes while retaining the unused sampled start in raw custody. Both
source branches require scalar outputs, uniform feature selection, shared initial
intervals and positive minimum depth. Classification requires explicit class
controls; regression rejects unused class controls.

`r0-extension-v1` additionally enables independent vector Gaussian leaves,
feature-specific intervals, weighted feature choices (including zero weight),
IID class leaves and depth-zero stumps. Majority ties can select the smallest
class, largest class or first tree's tied label. Class outputs are actual int64
labels; they are not regression values silently bucketed into categories.
Regression returns a float64 matrix and averages each output coordinate.

`RecursiveTreeNode`, `RecursiveTree`, and `RecursiveForest` are authored immutable
records. Nodes retain sampled feature, threshold and interval even at leaves,
plus depth, child indices and exactly one leaf value/label representation.
Validation rejects cycles, sharing, unreachable nodes, invalid depths, out-of-range
features/classes and inconsistent repeated-feature ancestor intervals.
Traversal stays float64; it does not inherit sklearn/XGBoost float32 casting.
Finite queries outside the construction interval are routed normally.

`sample_recursive_forest` returns `SampledRecursiveForest`: plan, ordered forests,
initial/final NumPy snapshots, complete raw draws, reconstruction identity and
caller limits. One injected stream owns all construction draws. Transactions
publish the caller's advanced state only after successful construction and replay
validation. All five supported NumPy engines work; no global RNG is used.

`frozen-function` stores one forest reusable on any admitted query matrix.
`fresh-forest-per-row` stores one forest for each original population row and
requires that exact row count on evaluation. It is a stochastic reference
population, not one frozen row-wise function. Reordering inputs alone changes the
forest/input binding; future graph adapters must preserve original row identity
and explicit matched-intervention ownership. The core does not claim that adapter
or intervention workflow is delivered yet.

`evaluate_recursive_forest` returns values; `inspect_recursive_forest` returns
visited node indices by original row, tree and path. Authored and sampled strict
JSON decoders are separate. Sampled reopening regenerates all trees and raw draws
from the retained snapshot, then compares every complete field with primitive-type
identity. Reading these archives neither fits a model nor imports an optional
backend. Old recipe selectors and old frozen-function versions are unchanged.

## Admission and verification scope

Before snapshots or draws, admission charges the worst-case
`populations * estimators * (2**(max_depth+1)-1)` nodes, full raw custody, path copies,
record/replay storage and RNG snapshots. Evaluation also charges input finite-check
masks, path traces and output buffers before any row-sized array. Caller and retained
limits intersect; insufficient limits reject rather than truncate, resample or
silently reduce a forest. These conservative estimates are not measured RSS or
Cartesian-product capacity claims.

Public tests use a separately written recursive literal oracle against the
iterative builder, including complete node records and final RNG states. They
cover all five engines, both class-start variants, depth extremes, exact-threshold
routing, sub-float32 distinctions, tie policies, weighted/vector/IID extensions,
per-row custody, archive tamper, strict numeric types, early admission and late
failure rollback. Author evidence is ignored under
`docs/evidence/issue-28-mitra-recursive-forest/`. Independent review and integrated
release checks remain separate from these author checks.
