# Sampled parameters composed with fitted populations

The [recorded parameter-law sampler](issue-28-fitted-parameter-laws.md) now also
composes with [explicit fitting lifecycles](issue-28-fitted-population-lifecycles.md).
These are combinations, not aliases between independent calibration and support
or batch fitting. Existing independent-calibration records and RNG schedules stay
unchanged.

`PopulationTreeParameterPrior` combines a `PopulationTreePlan`, named scalar laws,
source profile, raw-draw limits and fitting population. It shares the validated
law catalog with the original sampler: fixed, categorical with optional weights,
inclusive integer-uniform, uniform, log-uniform and capped-exponential. Family,
depth, estimator count, class count, splitter, classifier rule and learning rate
remain subject to backend/task compatibility. DT/RF/ExtraTrees/GBDT and regression
or classification are available; varying a parameter that some selected backend
ignores is rejected. The one exception is the pinned source’s DT estimator-count
draw, which is consumed, retained and explicitly labelled unused.

| Lifecycle | Effective fitting row count | Parameters and data streams |
| --- | --- | --- |
| Existing independent calibration | Existing independently selected fit size | Existing five-stream graph API unchanged |
| Support-only | Actual context-row count, including one row | Separate parameter/input/target/backend streams; query rows never enter fitting |
| Batch-refit reference | Actual whole episode size | Explicit new fit event retains its own parameters, fake targets and backend state |
| Mixed independent population | Fixed or sampled size, 1–1024 rows | Separate synthetic population, including recorded Gaussian/categorical options |

Support/batch `rows` laws reject **before parameter draws**, even when a law would
always return the correct size. Sampling and then discarding a fit-size draw
would falsely advertise a varying dimension. In shared construction controls,
omit `prior.base.rows` to bind it to the actual context/episode; an explicit
conflicting count rejects. The archived prior contains the effective count.
Mixed-independent fitting accepts fit-size laws, including one-row draws. A
realized one-class boosted classifier still fails without inserting classes or
retrying.

## Source profiles and exact custody

The separately named pinned profile retains
`depth=min(4,2+floor(Exp(scale=2)))` and
`estimators=min(4,1+floor(Exp(scale=2)))`, including capped tail mass and source
draw consumption. Its source is the
[audited TabICL pin](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py).
Using those two laws in an O’Prior-inspired support lifecycle, Mitra-inspired
mixed population, classifier mechanism or sklearn GBDT is an explicitly labelled
r0 composition. It is not evidence that the paper used that combination.

`SampledPopulationTreePlan` separately retains parameter-source identity,
effective-row binding, lifecycle adaptation, backend semantics, selected plan,
initial/final parameter snapshots, raw draws and unused source-only choices.
`SampledPopulationFittedGraph` binds every selected plan exactly to the actual
fitted backend plan and requires contiguous parameter-stream states across nodes.
Fit-data streams retain their existing separate custody. Backend-free decoding
replays the parameter laws and checks source/row/lifecycle labels as well as fit
tables, ancestor populations and frozen equations.

All supported option extrema and categorical backend combinations are admitted
before drawing. Aggregate bounds include existing graph metadata, worst-case fit
tables and tree geometry, parameter raw records and snapshots. A single outer
transaction owns all four injected streams; downstream fitting or validation
failure rolls every stream back. There is no redraw-until-success behavior.

## Explorer and shared notebook controls

The shared explorer adds **54 composed presets**: the three new populations,
four named backends, both task types, and explicit versus pinned-law profiles,
plus six weighted backend-mixture presets. They supplement existing independent
calibration and explicit population presets. JSON exposes each base plan and law
without inventing source distributions for undisclosed settings. Presets vary
only compatible parameters; all selected dimensions are visible.

Construction binds the actual episode seed/task ID, uses four separate streams,
and runs inside the existing bounded worker. The active dataset shows sampled
plans, effective rows, parameter source, lifecycle/backend labels, raw counts
and unused pinned DT count. Next/replay preserve the complete frozen record.
Original-row inspection, mechanism edits, linked save/reopen, ZIP export and the
local notebook callback use the same records. Support-mode authored edits retain
the separate support/query sampling contract.

## Remaining scope and verification boundary

Full TabICL architecture/noise/backend reproduction, undisclosed O’Prior settings,
standalone Mitra generation and unavailable XGBoost remain separate requirements.
This change does not make a full-paper or scientific-quality claim. Hosted Colab,
GPU and maximum-resource qualification remain **NOT_RUN**. Root owns the combined
full suite; author and independent frozen-head verification remain distinct.
Gitignored receipts are under `docs/evidence/issue-28-sampled-population-fits/`.
