# Offline comparison runner

Status: Approved by Piotr on 2026-11-24 after four interview rounds and final
shared-understanding confirmation. This document is the authoritative accepted
scope for #26.

Ticket: https://github.com/Avistian/r0/issues/26

## Accepted decisions

1. Deliver the runner, baseline adapters and offline verification using fixtures.
   Real benchmark execution remains gated on a separately approved protocol
   freeze. Refuse real execution against an unfrozen protocol.
2. Use scikit-learn ridge regression and logistic regression for linear
   baselines, and histogram gradient boosting for boosted-tree baselines.
   Freeze one configuration per task type across datasets, without tuning
   sweeps. The accepted recipes are recorded below.
3. Select `tabpfn==2.2.1`, bound to the recorded wheel and explicit local V2
   weight hashes. Disable telemetry. Record compatibility failures explicitly;
   changing the package requires a protocol revision. This decision does not
   authorize installation, asset downloads or real inference.
4. Fairness requires identical context/query rows, feature information, support
   counts and resource caps. Predictor-specific preprocessing is permitted
   when explicitly recorded and frozen. Fit learned preprocessing only on
   context rows; query rows must not contribute fitted statistics.
5. Linear preprocessing uses context-median numeric imputation, zero for an
   entirely missing context column, missingness indicators, standardized numeric
   features and one-hot categorical features. Distinguish missing from unseen
   categories. Regression targets remain in original units.
6. Tree preprocessing retains numeric missing values and uses native categorical
   features without scaling. Preserve missing versus unseen-category indicators.
   More than 255 context categories in a feature produces an unsupported result;
   do not silently group categories.
7. Freeze these baseline recipes without tuning:
   - Simple: context arithmetic mean or unsmoothed context class frequencies.
   - Ridge: alpha 1, intercept, SVD solver.
   - Logistic: L2, C 1, intercept, LBFGS, 1,000 iterations, tolerance 1e-4,
     no class weighting.
   - Histogram boosting: 100 iterations, learning rate 0.1, 15 leaves,
     minimum 10 samples per leaf, L2 1, no early stopping, seed 0.
   Record all resolved parameters and exact library versions. Convergence
   failure is a failed attempt, without automatic retuning.
8. Canonical row identity binds source-file hash, parser recipe and original
   zero-based data-row position. Assign identities before exclusions; never
   renumber surviving rows. Source-provided IDs are additional metadata.
9. Implement a new `r0.modules.comparison` module with a public runner interface
   and separate TabPFN command. Enforce training isolation through import rules;
   preserve the protected evaluation directory.
10. Preparation emits one immutable comparison manifest consumed by every
    predictor, including separately executed TabPFN. Bind protocol, input assets,
    schema, ordered query/context IDs and sampling recipe. Predictors never
    independently resample. Cache identity additionally binds predictor settings,
    preprocessing, weights and execution environment.
11. Offline verification covers regression, binary and multiclass classification
    using tiny authored fixtures, including missing values, unseen categories and
    reordered queries. Label fixture evidence explicitly. Check the real
    256-query/256-context dimensions separately using synthetic data.
12. Run TabPFN in a separately provisioned subprocess environment because its
    dependency requirements conflict with r0's. Preserve both sets of dependency
    requirements. Use the same host CPU and resource limits, record both exact
    environments, and exchange frozen predictor inputs and predictions through a
    versioned file contract.
13. Sample by deterministic SHA-256 ranking of a versioned encoding of seed and
    canonical row ID, breaking hash ties by row ID. For classification, allocate
    at least one row per declared class, then allocate remaining slots to minimize
    squared deviation from eligible-source proportions, breaking ties by declared
    class order. Respect available counts and reserve context capacity when
    selecting queries. Infeasibility blocks preparation; never search seeds.
14. Each attempt runs in a separate Linux cgroup with one CPU's compute allowance,
    one numerical-library thread, an 8-GiB memory limit, no swap and a five-minute
    wall deadline. Include preprocessing, loading, fitting and prediction.
    Terminate the entire process group on timeout; fail preflight if enforcement
    is unavailable. Local capability inspection is not enforcement verification.
15. Freeze these numerical checks:
    - Probability bounds and sum-to-one: absolute tolerance 1e-6, without clipping
      or renormalizing in scoring.
    - Query ordering/chunking invariance: atol 1e-6, rtol 1e-5, after restoring
      row order; regression predictions remain in original target units.
    - Float64 score recomputation: atol 1e-12, rtol 1e-10.
    Failed checks do not authorize automatic tolerance widening.
16. Classification adapters return probabilities in declared class order;
    regression adapters return one finite scalar per query. Include query IDs
    and reject missing, duplicate or extra rows. Accuracy argmax ties use
    declared class order. Query targets belong only to scoring. Record all
    adapter probability conversions explicitly.
17. Results are immutable attempt bundles. Reuse cached results only after exact
    identity and integrity validation. Retain predictions, scoring inputs,
    predictor views, fitted preprocessing evidence, versions, timing and failure
    reasons. A cache mismatch requires a fresh authorized attempt; preserve and
    report corruption. Missing required results make the comparison INCOMPLETE;
    preserve inspectable valid partial results.
18. Completion requires automated checks and a retained hand-worked walkthrough:
    trace a query across predictors, recompute classification and regression
    metrics, inspect context-only fitting, reject changed cache identity and
    invalid predictions. Verify timeout/memory enforcement with harmless fixture
    processes. Explicitly label all fixture TabPFN evidence.
19. Freeze the TabPFN recipe: eight estimators, seed 0, CPU float32, one
    preprocessing worker, `fit_preprocessors`, no context subsampling, temperature
    0.9, no probability balancing, and no averaging before softmax. Resolve and
    record the pinned release's preprocessing recipes explicitly. Return class
    probabilities or the regression predictive mean. Actual compatibility and
    query-invariance verification remain pending until an authorized real run;
    source inspection and fixtures cannot establish them.
20. Support a learner adapter without requiring a trained r0 checkpoint to finish
    fixture-backed implementation. Verify its interface with an explicitly
    labeled fixture predictor. A real comparison without a valid learner
    checkpoint is INCOMPLETE; never substitute fixture predictions.
21. Use versioned JSON manifests and hashed, non-pickle array files. Separate
    scoring targets from predictor inputs; give predictor subprocesses only
    their input bundle. Preserve numeric precision, category mappings and
    missingness explicitly. Keep real scoring evidence under owner custody;
    commit authored fixtures and permitted evidence. Bind implementation identity
    into cache identity as well as configuration and dependencies.
22. The implementer may choose mechanical details such as file layout, CLI
    spelling, serialization encoding and test organization, provided they are
    versioned, documented and obey this specification. Resolve unspecified
    library defaults once and record them before protocol freeze. Changes to
    algorithms, preprocessing semantics, tolerances, resource limits or scientific
    scope require another decision.

## Inherited protocol requirements

The inspected protocol packet and revised #25 specification are pinned here:

- [Protocol packet](https://github.com/Avistian/r0/blob/bd54aa576e1fe51e799bf5db35c490dd203ccbef/docs/protocols/single-table-v0/README.md)
- [Protocol specification](https://github.com/Avistian/r0/blob/bd54aa576e1fe51e799bf5db35c490dd203ccbef/docs/specs/0025-evaluation-protocol.md)

The revised #25 specification still requires re-acceptance, dataset admission
and human protocol freeze. Its closed issue state is not a freeze receipt.
This interview does not implicitly approve those outstanding decisions. The
following constraints carry into #26's implementation:

- Real episodes have 256 fixed query rows and three draws of 256 distinct
  context rows. Queries are disjoint from contexts; contexts may overlap each
  other. Freeze query seed 0 and context seeds 1, 2 and 3. Classification includes
  every declared class; regression samples uniformly without replacement.
- Classification primary score is Brier, summing all declared class columns
  including both binary columns; secondary score is accuracy. Regression scores
  are RMSE and MAE in original target units.
- Retain all three draw scores, their arithmetic mean and min/max. Keep task
  types and development/milestone partitions separate. Average draws within a
  dataset, then weight datasets equally within a task type. Never pool raw RMSE
  across different target units.
- TabPFN gaps are descriptive. Regression's normalized gap uses context-mean
  baseline RMSE as denominator; zero makes the gap undefined and a required
  comparison INCOMPLETE. Measurement completeness and learner quality are
  separate verdicts; retain valid negative results.
- Retain every failed attempt. Only an infrastructure failure permits one
  identical retry, with separate resource authorization. Timeouts, invalid
  predictions and poor scores do not qualify. Never change seeds/settings or
  choose the better score across attempts.
- Real execution requires applicable admission/freeze, learner-selection,
  access and budget receipts. Real TabPFN runs additionally require the guarded
  execution lane. Design approval does not authorize those runs or downloads.
- TabPFN outputs must not influence r0 training or model selection. Preserve
  milestone custody, existing exposure records and private challenge custody.
  Keep protected evaluation files and ADR 0003 untouched.

## Verification and remaining external gates

Implementation verification must distinguish automated fixture tests, retained
hand-worked review, resource-enforcement probes and actual-model evidence.
Include a fit-row spy for learned preprocessing and estimators, classification
and regression metric oracles, exact split/support checks, cache tampering,
invalid-output rejection, query-order/chunk invariance checks and an offline
smoke run. Record actual commands and results during implementation; none have
been run as part of this design interview.

Real TabPFN compatibility, resource feasibility and numerical invariance remain
NOT_RUN. Dataset admission, revised #25 acceptance and protocol freeze remain
PENDING. These external gates do not prevent fixture-backed #26 implementation
but must block a real comparison until satisfied.

## Handoff

Implementation is delivered through PR #68. Piotr explicitly requested detailed
accepted decisions in #26 after confirming this specification; preserve those
issue details. This published document is the authoritative accepted scope.
The issue's Approved specification section records the verified commit-pinned
revision. Real-run authorization and protocol freeze remain separate gates.
