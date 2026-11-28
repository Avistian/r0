# Generator branch CI timeout

GitHub run `46152933929` at source `826a6fc` passed formatting, lint, types,
boundaries and repository guards, then exceeded the check job's 15-minute limit
while running pytest. The GitHub annotation explicitly reports the execution
limit. This is distinct from run `46152406645`, which stopped at Markdown Python
formatting and was repaired in `826a6fc`.

The timed-out run reached 3% of pytest without an assertion failure being printed.
It is not a passing test suite; the remaining tests and walkthroughs did not run.
The retained log is gitignored at
`docs/evidence/issue-30-ci-repair/github-timeout-full.log`.

Two earlier full local suites finished successfully in roughly 69–70 minutes:
`dfa8b22` (5,650 passed) and `aa3ddda` (5,774 passed). A later local full suite at
`826a6fc` remains a separate live process. The existing timeout therefore cannot
be justified by those measured full-run durations. One representative causal
save/reopen/replay test completed in 2.88 seconds; constraining OpenMP/BLAS to one
thread took 3.15 seconds in a separate probe. This small comparison does not
establish a suite-wide threading cause or exclude every slow/hung test.

The check job allowance is now 120 minutes. All `make check` gates, tests and
subsequent generator walkthroughs remain enabled and ordered as before. No
dependency, test exclusion, retry or alternate passing criterion was added.
This change repairs the demonstrated budget mismatch; a completed hosted run is
still required to establish hosted acceptance. If the job exceeds the new cap,
investigate retained timings and partition work without dropping coverage.

The diagnosing-bugs workflow used the actual hosted timeout as its failing
reproduction and narrowed a representative session path for timing. The global
time-budget failure cannot be reproduced by one isolated test without changing
the symptom. The successful full-run measurements and hosted timeout annotation,
not the small timing probe alone, justify the workflow change. No synthetic test
asserting the YAML timeout value would establish that the suite actually finishes.
