# Issue 25 scoring and freeze walkthrough

This is a hand-authored comparison, not a learner/TabPFN run, a production-sized episode,
or proof that the harness enforces the protocol. The [accepted specification](../../specs/0025-evaluation-protocol.md)
and [candidate manifest](../../protocols/single-table-v0/README.md) define the actual milestone.
Human acceptance, real assets, overlap clearance, exact baseline pins and final freeze remain
PENDING. Runtime harness enforcement belongs to #26.

## Follow one query

[comparison.json](comparison.json) contains three tiny source tasks, three context draws per
task and two fixed queries per task. Each row has a source-scoped ID and one feature. All
values and predictions are authored literals. Each supplied predictor sees the same context
features/labels, query features and declared class vocabulary. scoring_targets is separate.
No fitted transformations are used in this arithmetic example (identity recipe).

For binary draw 1, query binary-q0 has feature [10], target 0 and candidate probabilities
[0.8, 0.2]. The target belongs only to scoring. The two context rows have labels 0 and 1,
so context-class-frequency probabilities are [0.5, 0.5]. Inspect query binary-q1 the same way.
Each context set is disjoint from the query IDs; all draws use the same query IDs. This file
illustrates the permitted views, not an executed isolation/spy test. #26 must prove that fit
sees context only and changing scoring targets cannot change predictions.

For multiclass, class-column order is [0, 1, 2], and each draw has one context label per
class. For regression, context means are 1, 2 and 3 respectively; each draw's baseline is
fitted to its own context. Candidate/reference predictions are supplied, not fitted here.
Accuracy ties choose the first declared class for this example; #26 must pin its convention.

## Independent worked scores

For binary draw 1, candidate Brier is
((0.8-1)^2 + 0.2^2 + 0.1^2 + (0.9-1)^2) / 2 = 0.05.
This sums both class columns, not the single-positive-column binary variant.
The reference has 0.125 and the context-frequency baseline 0.5 on every draw.

For multiclass draw 1, squared-error sums are 0.06 and 0.14: Brier = 0.10.
The reference has 0.14 on every draw, and the frequency baseline has 2/3.

| Type / primary score | Candidate draws | Reference draws | Simple baseline draws |
| --- | --- | --- | --- |
| Binary / Brier | 0.05, 0.25, 0.29 | 0.125, 0.125, 0.125 | 0.5, 0.5, 0.5 |
| Multiclass / Brier | 0.10, 0.24, 0.375 | 0.14, 0.14, 0.14 | 2/3, 2/3, 2/3 |
| Regression / RMSE | sqrt(1/2), 1, sqrt(1/2) | sqrt(2), sqrt(2), sqrt(2) | sqrt(5), 2, sqrt(5) |

All candidate/reference classification accuracies are 1. The binary simple baseline accuracy
is 1/2; multiclass simple baseline accuracy is 1/2 under the declared tie convention.
Regression candidate MAEs are 0.5, 1, 0.5; reference MAEs are 1, 1, 1; baseline MAEs are 2, 2, 2.

For regression draw 1, candidate RMSE = sqrt((0^2 + (3-4)^2)/2) = sqrt(1/2).
The context mean 1 predicts [1,1], so baseline RMSE = sqrt((1 + 9)/2) = sqrt(5).
The normalized candidate-reference gap is -1/sqrt(10), approximately -0.3162277660.
Draw 2 has gap (1-sqrt(2))/2, approximately -0.2071067812; draw 3 repeats draw 1's gap.
Normalize each draw before averaging; do not divide two already averaged RMSEs.

| Task-type summary (one dataset each) | Candidate mean [min, max] | Mean gap to supplied reference |
| --- | --- | --- |
| Binary | 59/300 = 0.1966666667 [0.05, 0.29] | 43/600 = +0.0716666667 |
| Multiclass | 143/600 = 0.2383333333 [0.10, 0.375] | 59/600 = +0.0983333333 |
| Regression | (sqrt(2)+1)/3 = 0.8047378541 [sqrt(1/2), 1] | (-2/sqrt(10)+(1-sqrt(2))/2)/3 = -0.2798541044 |

Each type has one authored task, so its equal-dataset-weight summary equals that task's
three-draw mean. If a type later has two datasets, average their dataset means equally,
regardless of query count. Never combine these three task-type summaries into one number.
The candidate beats the simple baseline on all three mean primary scores in this example,
but is worse than the supplied reference for both classification types. These are arithmetic
observations, not measured model-quality or benchmark verdicts. Draw ranges are not confidence
intervals; the same query rows recur across draws.

## Failure and access walkthrough

A reviewer can change the example on paper without modifying any retained run:

- Replace one required prediction with missing/nonfinite output: that attempt is invalid,
  retain the other scores, and mark the comparison INCOMPLETE. Never remove the query.
- Use constant regression targets/context with a perfect mean baseline: denominator zero,
  normalized gap UNDEFINED, required overall verdict INCOMPLETE, even if raw RMSE is valid.
- Replace a probability vector with [0.8, 0.8]: reject it rather than silently renormalize.
- Simulate a machine interruption before completion: one identical authorized infrastructure
  retry is allowed, with both attempt records retained. Timeout or bad prediction is not
  eligible; neither permits larger caps, new seeds or another checkpoint.
- Change a split or reference checkpoint: require a new protocol identity and run record.
  Keep old evidence and reject cross-identity cache/comparison reuse in #26.
- Reveal a milestone score: its holdout is no longer fresh. An unsuccessful result or a new
  manifest hash cannot reverse that exposure. Keep the private challenge contents outside Git.

These are expected outcomes for the future harness, not executed rejection/timeout tests.
The live checker/replay implementation, isolation tests and resource enforcement are #26 work.

## Verification and limits

The accompanying verification record reports local independent arithmetic and packet-identity
checks. They validate the authored evidence, not a model, the real assets or a frozen suite.
No production code/dependencies changed; no TDD feature claim is made. Repository checks and
independent spec/standards review cover this documentation delivery. Smoke training and eval
remain NOT_RUN: their CLI paths are still placeholders and outside this ticket's execution scope.

Human review: follow a row, recompute Brier and RMSE, trace each draw into its task summary,
then review candidate exclusions and the pending freeze checklist. Record the human receipt
outside this example; this walkthrough does not self-approve #25 or close it.
