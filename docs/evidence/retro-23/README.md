# #23 retrospective verification

Scope: activate existing local commit guards, share the 1 MiB artifact rule with CI,
and broaden public-boundary diagnostic corruption checks. No generators or scientific
acceptance thresholds changed.

Implementation reviewed: `47f73000a41a35a5ac6558bb19d7690cf5dd097f`, relative to
#55's `8e1a70a`. The new guard tests failed before the checker scripts existed, then
passed with the implementations. Existing #23 regression cases remain intact.

## Guard walkthrough

[Raw console output](guard-walkthrough.txt) records an actual disposable clone using
the existing environment and ordinary `git commit` calls:

- Local preflight failed before hook activation.
- `make hooks` configured the repository hook without syncing dependencies.
- Local preflight passed with the reused environment and correct source checkout.
- A commit containing an unused import failed with Ruff F401.
- A commit containing a staged 1,048,577-byte blob failed even after its working file
  was deleted; the same blob failed the CI-facing `make artifact-sizes` target.
- Both rejected commits left HEAD unchanged. The disposable clone was removed.

The guard tests additionally check executable-bit repair and acceptance at exactly
1,048,576 bytes. The size checker examines Git's index, not unstaged file contents.
CI skips local hook activation and runs the repository checks directly.

## Diagnostic coverage

The matrix covers six families × three task types × two noise regimes. Each valid
control passes. Missing streams, changed partitions, consistently corrupted targets
and corrupted latent traces produce their specified findings, count as invalid, and
are not selected. Tree leaf corruption is checked across every task/noise regime.

Malformed and nonfinite returned traces leave completed failure reports, all proposal
records, original returned-record evidence and manifests across all six families and
all three task types. These are failure-retention checks, not learner-quality evidence.

## Independent reviews

Standards: PASS, no findings at `47f7300`; public module seams and repository standards
respected. Spec: PASS, no findings; all three approved retrospective changes covered.
Each reviewer independently ran 59 focused tests successfully. Reviewers did not claim
full-suite, hosted-CI or normal-commit evidence; the latter is the author walkthrough
linked above. Full-suite and hosted-CI results are reported on the PR separately.

Author validation: `make check` passed with **138 tests**, formatting, lint, Pyright,
import boundaries, module/file guards, local hook preflight and the shared size guard.
The full console output is retained in [make-check.txt](make-check.txt).
