# Issue 28: actual-population XGBoost mechanisms

This additive core extends `I1.function.07`, `I1.code.09–10`, and
`OP.function.02`; it does not close those broad inventory items or issue 28.
The existing `XGBoostFitPlan` / `fit_xgboost` independent synthetic-input
construction remains unchanged.

## Source and named reconstruction

The pinned TabICL `TreeLayer.forward` fits XGBoost to the actual incoming
parent matrix and fresh artificial normal targets, then evaluates on those
inputs: [source lines 56–90](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py#L56-L90).
Its source parameter choices are separate from this lifecycle fact. This
implementation retains r0's explicit CPU histogram/vector-leaf parameter
plan and supported objective laws. It does not claim to reproduce every
source distribution or backend category-split option.

Three explicit lifecycles are available:

| Population | Actual fitting inputs | Subsequent generation |
| --- | --- | --- |
| `support-only` | Emitted episode's actual context rows of each node's parents, in context order | Frozen model, same seed/task identity only |
| `batch-refit-reference` | Every actual parent row in the emitted episode, in original row order | Frozen reference realization, same seed/task identity only |
| `independent-ancestors` | Separate same-authored-graph population with the retained fit seed/task and requested fit row count | Frozen model, new query seed/task permitted |

Support and independent modes are named r0 constructions. Batch mode exposes
the source's input-dependent fit as an explicit retained construction event;
it does not silently retrain while replaying or inspecting a query. A new
batch requires a new explicit `fit_xgboost_population_graph` call. All tables
are synthetic. There is no evaluation-data fitting, GPU or hosted execution.

## Public core

`PopulationXGBoostPlan` exposes the existing XGBoost controls with actual fit
row counts of 1–1024. `fit_population_xgboost` accepts the actual float64 input
matrix, a lifecycle, and independent injected target/backend generators. It
retains inputs, artificial targets, native JSON, backend parameters, exported
frozen trees, two initial/final snapshots, and both raw draw records.
Regression targets are normal; binary/multiclass targets use the existing
explicit integer laws. The backend is the existing XGBoost 3.4.1 dependency.

`fit_xgboost_population_graph` accepts an authored functional, native, vector,
or converted graph plus ordered node/plan pairs. It retains the original
graph. For each selected node it evaluates actual parents after preceding
selected mechanisms have been fitted, records the exact original fit row IDs
and prefix digest, and replaces only that structural function. Runtime node
noise, domains, converters, readouts, and target adapters stay with the
original graph. All selected nodes use one declared fit-population geometry.
Independent ancestor fits require at least 16 rows, inherited from the
ordinary authored task's minimum row geometry.

`XGBoostPopulationGraph`, `PopulationXGBoostNodeFit`,
`population_xgboost_fit_from_dict`, and `xgboost_population_graph_from_dict`
are public. Ordinary `generate_task`, `recipe_from_dict`, generation
`RunRequest`/producer validation, ZIP save/load, and `explain_generation`
recognize this wrapper. Inspection preserves ordinary nested graph stages
and repeated original query rows. Separate fitting-table nodes show retained
fit row labels and a fitting-population scope, independent of the query row
selection; the display is limited to the first 32 retained fit rows while the
archive retains every row.

No backend import or fitting occurs during model replay, graph generation,
archive decoding or inspection. Full task validation compares canonical
regeneration, execution metadata, identities and all tensors. Support/batch
run requests reject count greater than one and changed fit identities before
source restoration; independent mode can emit multiple new query tasks.

## Admission and custody

The construction path checks complete plans, graph geometry, fit population,
node widths, identities and conservative aggregate raw/table/model workspace
before fitting or RNG transactions. Prospective native model storage and
retained numeric records are bounded. Backend failure rolls back both
caller-owned streams across the entire sequential graph construction.

Raw graph decoding checks the complete outer envelope, original authored
metadata, every fit record, snapshots, raw declarations, row geometry, prefix
identity and stream chain before restoring any fit stream. It then replays
raw draws and verifies the actual synthetic parent tables against the retained
fit identity. Native JSON is checked against the exported frozen function and
plan geometry. There is no retry, substituted forest backend, hidden jitter,
row clipping, or changed fitting data on failure.

Native-model/export consistency and retained-data custody are validated.
Decoding does not refit the native backend to prove that arbitrary externally
authored model JSON was trained from the supplied data. Independent native
retraining is verification evidence, not an archive authenticity guarantee.

## Bounded author verification

Public tests cover actual-input native fitting, four graph families across
three lifecycles, sequential fitted ancestors, full JSON and ZIP replay,
large task IDs, repeated inspection rows, producer tensor custody, malformed
outer metadata before restore, resource rejection before backend work, and
transaction rollback. Genuine missing-API RED results and the inspection
metadata rejection RED are retained separately from GREEN results.

Outside pytest, 60 graph cases (four families × three lifecycles × five RNG
engines) compare actual parent tables and context row identities, independently
retrain native XGBoost, and traverse exported trees one row at a time. Graph
target errors are exactly zero. Another 15 cases (three native objectives ×
five engines) compare scalar traversal against native prediction. Replay is
instrumented to forbid hidden backend training. Twelve ZIPs are reopened and
fully validated. Nine malformed archives reject with zero snapshot restores;
a valid control restores four times. All five engines retain exact caller
states after a deliberate native training failure. Four legacy artifact
hashes (three ordinary task types and the original independent XGBoost fit)
match ancestor `a19adab1c805a0d0338479201109f0a6624eeb64`.

Raw scripts, terminal results, and artifact hashes are ignored under
`docs/evidence/issue-28-xgboost-population/`. These are author-run checks,
not independent review, a full suite, browser proof, or whole-option acceptance.

## Required follow-ons

The guarded construction workbench is the next separate child: full visible
lifecycle/plan controls, original-row graph plus fit-table inspection,
save/reopen/export/replay and explicit independently fitted versus batch
reference semantics. No browser support is claimed by this core.

Source parameter hyperpriors and the complete source backend option catalog
remain required. Population mechanisms inside sampled joint hyperpriors,
paired interventions, calibrated wrappers, references, and policy environments
need explicit composition support; this wrapper does not silently inherit
those APIs. Mixed Conv/network/parent equations are accepted by the existing
authored graph contracts, but exhaustive mixed-function and categorical
composition verification is still required. The four-family test grid is
representative coverage, not proof of every option or combination. Existing
issue 30 selection and issue 39 row-axis boundaries are unchanged.
