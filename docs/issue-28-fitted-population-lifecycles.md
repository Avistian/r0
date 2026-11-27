# Explicit fitted populations and lifecycles

This #28 slice adds three separately named fit events alongside the existing
independent frozen-ancestor calibration. Every event exports a frozen function;
archive replay, inspection, interventions and Next never call sklearn again.
Constructing another support or batch event fits a new function. Construction
streams are keyed by construction seed plus the actual episode seed/task ID, so
a new batch also receives new fake-target draws rather than reusing a previous
batch’s target vector. Batch-reference
mode is not a training-ready default and has no cross-batch function identity.

| Population | Inputs used to fit each node | Identity and source scope |
| --- | --- | --- |
| Existing independent calibration | Fresh synthetic ancestors, separate from episode rows | Existing records and RNG schedule unchanged |
| `support-only` | Actual context parent states, in declared context-index order | O’Prior-inspired support-dependence contract, explicit r0 fake targets/backend settings |
| `batch-refit-reference` | All actual episode parent states, including query rows | TabICL-inspired fit/predict-on-batch reference; each construction retains its own fit |
| `mixed-independent` | Separate Gaussian/categorical synthetic fit population | Mitra-inspired disclosed population laws with explicit r0 values for undisclosed settings |

Source references and exact distinctions are recorded in
[the source-law audit](research/0028-fitted-tree-source-laws.md).
[O’Prior v2](https://arxiv.org/html/2605.18971v2#S2) requires fitted statistics to
depend on support. It does not disclose all fake targets, backend parameters or
repeated-call freezing rules. The explicit fit-then-freeze API here is an r0
lifecycle choice. [Pinned TabICL](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py)
refits on incoming rows on each forward. This API makes that event explicit and
retains the resulting state; it does not reproduce the complete TabICL graph,
noise sequence, XGBoost multi-output backend or historical library defaults.
[Mitra v1 A.2](https://arxiv.org/html/2510.21204v1#A1.SS2) fits separate synthetic
tables. Its Gaussian parameters, geometric parameter, probability vectors and
fit-size choices are not fully disclosed. Recorded choices here are adaptations.

## APIs and broad options

`PopulationTreePlan` supports DT, ExtraTrees, RF and GBDT, regression and native
classifier mechanisms, multiple outputs, depth, estimator count, DT splitter,
class counts, probability averaging or explicit majority extension, and boosting
learning rate. The new plan admits **1–1024 actual rows**. Old `FittedTreePlan`
retains its 32-row minimum and old serialization. There is no padding, duplicate
row insertion, extra class insertion, retry or easy-case resampling.

`fit_population_tree` consumes actual selected inputs or a `MixedFitPopulation`.
`fit_population_graph` replaces selected nodes topologically. Downstream fit
inputs use already frozen upstream fitted mechanisms. Selected nodes cannot
reuse pre-fitted pipelines; other frozen ancestor mechanisms retain their usual
semantics. Classifier outputs are structural category codes, not automatically
the enclosing task’s labels or class probabilities.

`MixedFitPopulation` exposes uniform sampling of a categorical fraction or an
explicit fixed fraction, count `floor(p_cat * (d+1))`, uniform index selection
without replacement, geometric cardinalities on positive integers, categorical
sampling, Gaussian feature mean/scale and regression target mean/scale. Category
probabilities are uniform or sampled through an explicitly named symmetric
Dirichlet extension. Classification targets can use a declared class count or
sample a geometric count bounded by the declared class cap. They are sampled
directly, without quantile bucketing. A geometric draw above the resource cap
**fails**; it is not clipped into a different distribution. A fixed fraction of
one produces an invalid count under the disclosed formula and rejects rather
than silently replacing the formula. Default fractions use `[0,1)`.

XGBoost remains unavailable under the dependency constraint. No sklearn family is
silently advertised as XGBoost. Full standalone Mitra generators and source
architecture/noise schedules remain separate required breadth; applying a mixed
fit population inside an authored SCM is an r0 structural adaptation.

## Support identity and frozen edits

Support and query roots/innovations use separate keyed streams. Values are then
scattered into the ordinary declared random split’s original row indices. This
matters for laws such as ball roots, whose direction and radius draws would
otherwise shift when the total query count changed. With fixed support controls
and identities, appending query rows leaves support values and fitted functions
unchanged. Batch-reference mode uses the ordinary full-batch stream contract.

Frozen support graph replay retains this split-population sampling contract.
The mechanism editor removes fitting claims while preserving the support/query
draw contract in `SupportPopulationFunctionalRecipe`; changing an equation must
not accidentally change all original roots. Matched interventions keep the same
split-population streams and frozen fits. Fresh-population interventions remain
explicit and do not refit.

## Custody, admission and UI

Records retain selected plans, actual fit tables and fake targets, categorical
cardinalities, raw draws, three initial/final stream snapshots, backend seeds and
full backend parameters/version, source-contract labels, and exported split
tables. Graph records bind functions to ancestors and original fit seed/task ID.
Reopening checks raw derivation, stream continuity across nodes, actual selected
parent rows, backend-plan consistency and equation binding without importing or
fitting sklearn. Raw records describe our draws; backend-internal random draws
are represented by seeds/version, not claimed captured.

Admission checks actual fit sizes, worst-case tree geometry, raw metadata and
aggregate graph working estimates before fit-table allocation. Failures roll
back the injected input, target and backend streams. Single-class boosted
classifiers reject explicitly; other backends may validly emit constant trees.
These engineering caps are not measured peak-RSS guarantees.

The shared explorer offers 24 population presets: all three new populations,
four backend families and both task types, plus the existing independent and
sampled-parameter presets. JSON controls expose the full options. Actual realized
plans, population, source/adaptation labels, cardinalities and raw-draw counts are
visible. Support/batch plan `rows` may be omitted to use the actual selected
population; an explicit conflicting value rejects. Construct is guarded by the
existing bounded worker. Next retains the frozen record and exact integer IDs.
Replay, original-row trace, mechanism editing, linked save/reopen and ZIP export
use the same local/browser/Colab callback. Hosted Colab remains **NOT_RUN**.

Verification receipts, including historical failures, are gitignored under
`docs/evidence/issue-28-fitted-lifecycles/`. Author checks are distinct from the
independent frozen-head review and root-owned combined full-suite verification.
