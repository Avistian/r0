# Issue 24 independent synthetic challenges

Status: four interview rounds accepted; implementation authorized with `$implement 24`
in small readable PRs. Public implementation is distinct from final ticket acceptance.

## Agreed scope

Four paired cases: column permutation, context-dependent three-class label mapping,
affine regression extrapolation, and a privately authored nuisance-association reversal.
Three pairs are public. The owner authors exact nuisance values and expected answers
outside development access, freezes their identity and retains private acceptance evidence.
No development agent constructs or reads that sealed case.

All predictors receive the same declared rule family and schema. Coefficients, mappings,
relevant-feature identities and query targets stay outside prediction. Context must identify
the answer within the declared family. No guessing an unidentifiable hidden rule.

Column permutations move both context/query values and schema; answers stay unchanged.
Three-class remapping includes all classes in context and changes query answers consistently.
Extrapolation fixes context and the affine rule, moving queries from inside to outside the
context range; expected answers follow that rule. Nuisance reversal fixes context and the
true rule, reversing a misleading query association; context must identify the true signal.

Reuse episode/replay interfaces, but independently author mechanisms and expected-answer
calculations. Never call production generator helpers to construct challenges or their answers.
A context-using reference solves each pair; fixed-position, fixed-label, range-clipping and
nuisance-following shortcuts fail their respective controls. Reference success alone is not
independent verification: retain separately worked expected answers.

## Bounds and evidence

Fixed authored values, no seed sweeps. Each episode has at most 12 context rows, 6 queries,
4 features. CPU only, no paid compute, no training or TabPFN calls, no new dependencies.
The command has a five-minute cap; the acceptance walkthrough has a 30-minute budget.
These bounds do not constrain future production generators or models.

Exact classification answers; regression comparisons use atol=1e-10 and rtol=1e-10.
Retain raw inputs, actual predictor views, declared assumptions, predictions, expected answers,
worked calculations, differences and expected shortcut failures. Report per pair, never an
aggregate challenge score. Retain failure reports. Records are immutable and replayable.

Within an episode context/query row identities are disjoint. Across paired episodes, shared
rows are deliberate: retain source identity, episode identity and transformation relationships.
Related cases remain together across diagnostic/sealed assignments.

Record development influence and familiarity against each identified frozen training mixture.
Independent code does not establish unseen mechanisms. Until comparison exists, novelty is
UNKNOWN. Label later results as familiar draws, unseen combinations or unseen mechanisms.
If all mechanisms enter training, require a fresh reserved challenge before unseen-mechanism
claims; do not permanently prohibit a training family.

## Custody and completion

The owner privately verifies the nuisance case and records its frozen identity, independent
calculations, reference success and expected shortcut failures. Publish a receipt without
case contents. After one frozen-milestone evaluation, retire its sealed status. Repeated
results are not fresh held-out evidence. Do not claim concealment for developer-authored cases.

Close #24 only after three public walkthroughs plus the owner's private verification receipt.
Until then sealed acceptance and overall ticket acceptance remain PENDING. Public success is
suite correctness, not learned-model quality. #25 owns milestone thresholds and protocol;
#26 owns benchmark integration. Protected evaluation paths remain untouched.

Review one public pair per weekly session, then reserve a session for private authorship and
verification within the owner's 2–3 spare hours/week. No trained-model evaluation is required
to close #24. No live tracking integration is included.

## Confirmed test seams and delivery

The user confirmed tests at the public challenge-suite entry point (known answers,
transformations, shortcut controls, query-label isolation) and replay/CLI boundary (retained
reports, bounded execution, explicit sealed PENDING). These do not certify the private case.

Deliver a stacked series: accepted specification; public challenge pairs and controls;
bounded replay command and reports; walkthrough evidence and private-custody handoff.
Keep each PR reviewable and use references to #24, not an automatic closing keyword.
