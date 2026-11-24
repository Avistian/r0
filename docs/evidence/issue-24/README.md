# Independent challenge walkthrough

Public suite implementation: `d921cf7c8000bfc7629961dcbcbf8d8eec335330`.
Accepted scope: [issue 24 specification](../../specs/0024-challenges.md).
The retained [public report manifest](public-report-manifest.json) and its three linked pair files contain raw episodes, actual predictor
views, answers, calculations, per-query differences, transformation checks and source hashes.
Public checks PASS. Sealed verification and overall issue acceptance remain PENDING.
Downstream learner usefulness is NOT_MEASURED; novelty against a training mixture is UNKNOWN.

## Replay

From this checkout with its existing environment:

```sh
PYTHONPATH=src uv run r0 challenges --out /tmp/r0-public-challenges.json
```

The fixed command has no configuration or private-input option. It runs exactly three pairs
on CPU and uses a POSIX 300-second process alarm. Non-POSIX execution fails explicitly.
The direct Python report function is deterministic but has no process deadline; use the CLI
for the capped command. No data download, paid compute, training or TabPFN call is involved.

Output paths are immutable: identical contents may be reused; different contents fail without
overwriting the old record. Source changes alter the source fingerprints and therefore require
a new output path. Raw coordinates are used; there is no fitted preprocessing. Python version,
source hashes and the versioned literal cases identify this replay. The content fingerprint is
computed with its own field set to the empty string. Incidental paths and timestamps are absent.

The retained report is split into a manifest and one file per pair to respect repository
file-size limits. To reconstruct the original report, replace `pair_files` in the manifest
with `pairs`, loading those JSON files in the listed order. Its content fingerprint then
matches the CLI output. The split is storage-only and does not alter any values.

## Inspect one row through each pair

All predictors receive a PredictorView plus the same declared family. Source IDs, expected
query answers and rule derivations appear only in the report, not the predictor view.

| Pair | Context deduction | Query answers before → after | Shortcut before → after |
| --- | --- | --- | --- |
| Column permutation | Rows cover all four binary combinations; only a determines the label: 0=low, 1=high | low, high → low, high | low, high → high, low |
| Context remapping | Match A/B/C to their context labels; all three classes are present | red, green, blue → blue, red, green | red, green, blue → red, green, blue |
| Affine extrapolation | (0,1), (1,3), (2,5) identify slope 2 and intercept 1 | 2, 4 → -3, 7 | 2, 4 → 1, 5 |

For permutation query q0, original values are (0,1) with schema (a,b). After permutation they
are (1,0) with schema (b,a). The uniquely context-identified feature still has value 0, so its
answer remains low. The fixed-position control reads the first value and incorrectly says high.

For remapping q0, the symbol remains A. Its context label changes from red to blue, therefore
its query answer must also change. The fixed-label control ignores that evidence and says red.

For extrapolation q0, the context is unchanged. Its input moves from 0.5 to -2. Calculate
2 × 0.5 + 1 = 2 and 2 × (-2) + 1 = -3. Clipping to the observed target range [1,5] gives 1,
which is wrong. The other query moves from 1.5 to 3, giving 4 then 7; clipping gives 5.

Classification checks are exact. Regression uses atol=rtol=1e-10. Each shortcut must solve
the before case and fail at least one after answer; each reference must solve both sides.
No aggregate quality score is computed. These tiny tasks are not statistical performance claims.

Check logs retain command output with trailing whitespace removed.

## Measured validation

- Specification head `42ef7cb`: make check PASS, 138 tests ([log](spec-check.log)).
- Public-pairs head `31e905a`: make check PASS, 143 tests ([log](pairs-check.log)).
- Corrected implementation `d921cf7` with these evidence files staged: make check PASS,
  150 tests, all five import contracts kept ([log](check.log)).
- Actual CLI replay, including Python startup: 0.14 seconds wall time on this host,
  measured with /usr/bin/time. This is an observation, not a portable performance guarantee.
- Hosted CI is not represented by these local checks.

## Verification history

TDD proceeded through public seams: first permutation import failed, then remapping and
affine tests failed with absent pairs. Each corresponding implementation made its test pass.
The report import initially failed; implementing report generation made replay/immutability
checks pass. The CLI initially rejected challenges; the added command passed quiet-output checks.

Independent review at `568aa31` found context ambiguity could be resolved by query support,
missing family delivery to controls, and lost invalid-case evidence. New tests reproduced the
first two defects (no exception; missing argument) and the invalid-case exception. Correction
`d921cf7` makes them pass and retains raw failed cases and earlier successful comparisons.
The public [failure-evidence script](failure_evidence.py) reproduces a wrong-answer failure and
an invalid-context failure through public interfaces. Their reports are retained beside this file.

The timeout test blocks output on a FIFO and sends SIGALRM. It confirms the installed handler
returns an error without a false PASS; it does not wait the full 300 seconds. An output failure
or timeout cannot guarantee a persisted report; stderr remains the execution failure evidence.

Standards review of `568aa31` found no blocking violation, with a nonblocking concern that replay
hashes challenge source files through package layout. Review of `568aa31..d921cf7` found no new
standards violation. The spec reviewer independently recomputed all answers, then confirmed the
four implementation findings resolved at `d921cf7` using inspection and public-interface probes.
These were bounded public-code reviews, not private-case or overall ticket acceptance.

Training and evaluation smoke are NOT_RUN: their train/eval CLI capabilities remain unimplemented.
The existing Make targets are placeholders. No loss summary or benchmark table is claimed.

## Custodian handoff — still pending

Reserve one separate weekly session to author and verify the nuisance pair privately. Keep exact
values, rule parameters, expected answers, scripts and private reports outside development access.
Context must uniquely identify the true signal; keep it fixed and reverse the misleading nuisance
association only in queries. True answers stay unchanged. Apply the same family/schema information,
row/feature caps and numerical policy as above. Independently calculate answers and verify that a
context-using reference solves both sides while a nuisance-following control fails after reversal.

Before use, freeze the private artifact identity and retain your calculations and predictions.
Publish only a receipt with: case/category and version; private artifact SHA-256; custodian;
verification date; reference/shortcut/transformation verdicts; independent-calculation verdict;
bounds checked; development exposure/use status; and milestone/model/mixture identities when used.
Do not put private values, seeds, mechanisms or predictions in that receipt. Do not send private
contents to a development agent for verification. No receipt is currently supplied or simulated.

At a frozen milestone, evaluate once, log its influence, then retire sealed status. Later reuse is
not fresh held-out evidence. Compare novelty with the actual identified mixture; separate familiar
draws, unseen combinations and unseen mechanisms. If the training portfolio absorbs all mechanisms,
reserve a fresh challenge before making unseen-mechanism claims. #25 owns those milestone criteria;
#26 owns harness integration. Close #24 only after the three public walkthroughs are accepted and the owner supplies the
private verification receipt. The written walkthrough is review material, not a claim that
the owner has completed review. Budget at most 30 minutes for the acceptance walkthrough;
review one public pair per weekly session if that better fits the 2–3 hours/week allowance.
