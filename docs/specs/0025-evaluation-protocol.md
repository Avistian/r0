# Issue 25 single-table evaluation protocol

Authoritative accepted scope from eight interview rounds and the owner's instruction to
implement #25 in small readable PRs. Design accepted; asset verification and human freeze
remain PENDING. This document does not authorize compute, downloads, merges or benchmark
access. Issue: <https://github.com/Avistian/r0/issues/25>.

## Deliverable and ownership

Deliver a candidate manifest, explicit scoring/sampling rules, a hand-worked comparison and
a human freeze handoff. #26 implements the offline harness, adapters and resource enforcement;
#34 owns operational run/budget records. #25 stays open until provisioned asset identities,
eligibility, overlap checks and the protocol receive human acceptance. Relational freeze is
#41. No trained model, live TabPFN run or paid compute is required to complete #25.

Protected evaluation files and ADR 0003 require human handling under AGENTS.md; leave them
untouched and identify this handoff in the PR. Publication of this specification is not the
human freeze. Preserve its accepted scope in one document and link a published revision from
#25's Approved specification section. Never merge automatically.

## Dataset roles and admission

Use six real datasets, one development and one milestone dataset for each task type:

| Task type | Development | Milestone |
| --- | --- | --- |
| Binary | Wisconsin Diagnostic Breast Cancer, diagnosis | Indian Liver Patient, liver-disease label |
| Multiclass | Contraceptive Method Choice, contraceptive method | Dry Bean, bean variety |
| Regression | Abalone, rings as a continuous target | Facial/oral temperature v1.0.0, aveOralM |

These are provisional candidates, not verified assets. Use only FLIR_groups1and2.csv for
thermography, one row per subject. Exclude aveOralF, other oral measurements, subject/date
identifiers and target-derived fields; never concatenate overlapping cameras/group files.
Exclude record identifiers and target-derived fields from all predictors. Record the exact
feature allowlist and source target vocabulary before sampling. Abalone's target is rings,
not an unrecorded conversion to age.

Require at least 512 eligible rows after exclusions. Review collection methods, identifiers
and known repeated measurements; reject tasks requiring temporal or entity-group splits.
Document residual uncertainty rather than claim proof of statistical independence. Keep
versions, derivatives and common-source datasets in the same development/milestone partition.
Exclude known TabArena overlaps and derivatives; unresolved source-family overlap blocks
admission. Audit names, aliases, shapes, source families and available content hashes. Unknown
TabPFN pretraining/development exposure is allowed with UNKNOWN provenance, never an unseen
claim. Hosting on the same repository alone does not mean a common collection family.

Owner provisions data/weights outside Git. Verify exact versions, hashes, licenses and
embedded notices; commit provenance and permitted evidence, not real datasets or weights.
Record excluded row identities/reasons and before/after support counts. Retain one canonical
copy of identical feature-and-target records. Quarantine identical-feature groups with
conflicting targets. These exclusions change the population and must be disclosed. Preserve
missing features; exclude missing, invalid or nonfinite targets. Do not silently replace a
candidate or relax admission after seeing performance: return a failed candidate for review.

## Episodes and access

Each dataset has exactly 256 fixed query rows disjoint from the context pool. Draw exactly
256 distinct labeled context rows three times, with frozen explicit seeds; contexts may
overlap each other. All predictors use identical row selections. Classification queries and
each context draw preserve eligible-source proportions as closely as integer counts permit,
with every declared class represented. Freeze allocation/tie-breaking and sampler versions;
infeasibility blocks admission, not an undocumented resample. Regression sampling is uniform
without replacement. Freeze canonical row identities, query IDs, each context's IDs and seeds,
so inspection does not depend on reconstructing a hidden sampler.

Predictors see context features/labels, query features and declared inference-available schema.
Query targets stay in the scoring view. Use context-only fitted transformations; each predictor
may have a different explicit, frozen recipe. No extra source information or fitting on query
rows. Query order and chunk size must preserve predictions within a declared numerical tolerance.
Normal probability conversion belongs to the predictor adapter, not silent scoring repair.

Selection uses development evidence only. Freeze the learner checkpoint(s), training-mixture
identity and inference settings before milestone access. Owner-controlled scoring assets keep
milestone query labels/scores outside development; use schema/provenance and authored fixtures
there. Public data is not secret: document project access/influence rather than claim concealment.
One final evaluation of the frozen selection retires that holdout from fresh held-out use.
A new protocol identity never restores secrecy. A valid negative result remains useful evidence.

## Predictors and inference allowance

Pin TabPFN v2 explicitly, including package version, weight filenames/revisions/hashes and
terms; never rely on an upstream default. Claims are about TabPFN v2, not current TabPFN.
Use mean/class-frequency, linear and boosted-tree baselines through #26, plus the learner.
Freeze one baseline configuration per task type across datasets; development may verify
compatibility, but no baseline hyperparameter sweeps. Claims concern these configurations,
not each algorithm's best achievable performance. Exact recipes/pins are freeze prerequisites.

All predictors run on the same recorded CPU environment. Per predictor/dataset/context draw:
five minutes and 8 GiB cover preprocessing, model loading, fitting and prediction. Record
scoring/artifact-writing costs separately. Record CPU model, thread limits, software environment,
measurement boundary and enforcement method before freeze. Caps are requirements, not measured
feasibility or a claim that current tooling enforces them. No automatic budget escalation.
Future TabPFN execution still uses the guarded lane and separate authorization; this protocol
is not a spending approval. Total run reservations/retries require the operational budget gate.

## Scores, aggregation and quality

Score queries only. Classification primary: Brier = mean over queries of the sum over declared
classes of (probability minus one-hot target)^2. Lower is better; secondary accuracy is higher
better. This convention includes both columns for binary classification. Regression primary:
RMSE in original target units; secondary MAE. Both lower is better. Predictions/targets and
class-column order must be retained. Reject missing/nonfinite predictions, invalid dimensions,
wrong labels and invalid probability vectors, allowing only a declared floating-point tolerance.
Do not silently clip, renormalize or drop queries in the scorer.

For each classification draw, gap = learner Brier - TabPFN Brier. For each regression draw,
normalized gap = (learner RMSE - TabPFN RMSE) / context-mean baseline RMSE. Also retain raw
RMSE and raw gaps. A zero denominator makes the normalized gap undefined, not zero or infinity.
Average the three draw-level scores/gaps within a dataset, then weight datasets equally within
each task type. Report binary, multiclass and regression separately, and development separately
from milestone. No pooled cross-type score, no averaging raw RMSE across different units.
Initially each partition has one dataset per type, so its summary is that dataset's result.

Report all three draws, arithmetic means and min/max ranges. No population-level significance
claim, dataset-level uncertainty claim or automatic broad superiority conclusion. Draws measure
context sensitivity on the same queries, not independent dataset replication.

Quality target: mean primary score strictly beats context-class-frequency prediction for each
classification type and context-mean prediction for regression. Equality does not beat it.
Report TabPFN gaps descriptively; no numerical closeness margin. Complete valid measurement and
quality-target attainment are separate verdicts. Missing the target does not prevent completion
of measurement or authorize tuning against milestone results.

Report #24's synthetic challenges per pair, separately from real-data results, with no combined
score or mandatory learned-model challenge release gate. Preserve novelty UNKNOWN until training
mixture comparison and private owner custody. Reference exact-answer tolerances are not learner
quality thresholds; a closed ticket alone does not establish a private acceptance receipt.

## Failures, identity and evidence

Any missing required result or undefined required summary makes the overall comparison
INCOMPLETE. Retain valid per-task scores and support counts; no silent exclusions or partial
success verdict. Allow one identical retry only for an infrastructure failure, within separately
authorized resources. Preserve both attempts and the failure classification. Timeouts, invalid
predictions and poor scores do not qualify. No new seeds/settings/checkpoints on retry; do not
choose the better score across attempts. Exposed milestone outcomes remain exposed after failure.

Retain row IDs, scoring inputs, raw predictions, actual predictor views, fitted-recipe identity,
checkpoint/package/weight identities, protocol identity, environment, timing and all failures.
Publish what asset terms allow; retain restricted evidence under owner custody for verification.
Never publish sealed challenge contents. Run records are immutable. Score-affecting changes to
data, splits, preprocessing, metrics, assets or inference settings require a new protocol identity;
format-only changes may preserve it. A configuration change affecting results also needs a new run.

## Human freeze and weekly review

Review one dataset per 2–3-hour weekly session. Before freeze, the owner accepts provenance,
overlap findings, exclusions/support, exact assets/recipes, row selections/seeds, resource settings,
query-invariance/probability tolerances, protocol identity and the worked scoring example.
Unresolved fields stay PENDING; never represent a candidate-manifest digest as an accepted freeze.
#26 may use this design for fixtures, but the real milestone is blocked until acceptance.
No deployment, tracking integration, training or model-quality certification belongs to this work.
