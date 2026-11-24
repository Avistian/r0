# PR 68 completion

Historical API and evidence. For the subsequent interface migration, see
[comparison simplification](../comparison-simplification/README.md).

The accepted scope is [the specification](../../specs/0026-offline-comparison.md).
The original [implementation receipt](fixture-receipt.json) and review RED logs
remain historical evidence. This delivery finishes fixture-backed implementation;
it does not admit real datasets, freeze the protocol or authorize real TabPFN runs.

## Behavior delivered

- All backends validate context/query separation before fitting, validate probability
  vectors, and reject nonfinite computed scores. Invariance checks reuse one fitted
  estimator, including learner backends executed in a separate process.
- `external_attempt` accepts an explicit argument vector, provisioned interpreter,
  hash-bound implementation/identity/assets, and a complete expected runtime identity.
  The same immutable attempt lifecycle scores and retains its outputs. A separate
  backend gets an input path, never a scoring path. Runtime/identity, fit IDs and
  invariance receipts must match; cache reads repeat those checks.
- `compare` accepts `external_predictors` and optionally `saved_attempts` (three paths
  per role). It verifies saved results against the expected preparation, predictor,
  draw and environment before inclusion. Fixture reports include descriptive gaps,
  quality-target verdicts and task/partition groups. Overall real status stays
  INCOMPLETE, with every fixture measurement labeled FIXTURE_ONLY.
- The real TabPFN loading backend verifies local wheel and weight hashes and compares
  installed package files with the wheel before importing the model package. It
  pins CPU float32, one preprocessing worker and Torch thread settings, eight
  estimators, seed 0 and the resolved 2.2.1 preprocessing/ensemble configuration.
  Regression uses predictive mean; probability columns follow declared class order.
  The library is not installed in the r0 environment, and no weights were downloaded
  or executed. [Source inspection receipt](tabpfn-source.json) binds the release.

## Reproduce fixture verification

Use an existing provisioned environment. Commands below reuse it deliberately and
never sync dependencies or rewrite editable-install metadata:

```sh
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/comparison
python3 scripts/run_check.py python scripts/comparison_external_walkthrough.py \
  --out /tmp/r0-comparison-external-fresh
```

The latter runs authored mean/frequency backends in actual separate Python processes
for binary, multiclass and regression tasks. It runs the TabPFN *role* separately,
then ingests its saved attempts alongside the learner role and baseline results.
These are substitutes, not actual TabPFN or learned-model predictions. Their gap
is zero and they do not beat the simple baseline. [Retained summaries](external-summary.json)
are from the enforced run, not the unenforced diagnostic command above.

CLI equivalents use `python -m r0.modules.comparison`:

- `run --external learner.json --external tabpfn.json --saved saved.json ...`
- `tabpfn --descriptor tabpfn.json --prepared PREPARED --draw 1 --out ATTEMPT
  --cgroup-root DELEGATED_ROOT`
- `environment --out runtime.json` in each provisioned environment, with the frozen
  numerical-library thread variables set to one.

Descriptors have `name` (learner/tabpfn), `command` (argument list beginning with
an absolute interpreter), `identity` and `files` (absolute paths to SHA-256 values).
Identity contains implementation, full environment, settings, checkpoint and
explicit evidence classification. Only authored fixture execution is admitted by
this implementation. Files include the worker script/config and identity file;
never place scoring targets in them. `saved.json` maps each role to ordered draw
1/2/3 attempt paths. Missing or incompatible receipts cannot produce a complete
comparison. The TabPFN worker command is `tabpfn-worker --config CONFIG`, with input
and output arguments supplied by the launcher. Its config records explicit local
wheel/model paths and the identity; settings/checkpoint must match the pinned
recipe. Invoking an actual model still belongs to the separately authorized guarded
lane; this fixture handoff does not authorize provisioning or execution.

## Kernel enforcement and the discovered defect

The first live probe exposed a defect in the submitted cgroup bootstrap. A buffered
write to `cgroup.procs` lost a permission error during implicit file close; the child
continued in `/init.scope`. Consequently a 128-MiB allocation succeeded under a
claimed 64-MiB cap. [The failed probe](resource-before-fix.json) is retained.

The corrected bootstrap uses an explicit unbuffered write and verifies child
membership before exec. Starting outside the delegated subtree now fails before
predictor execution. Run inside a dedicated delegated service so migration is
permitted; never use or move another service's processes.

From the repository root, with an absolute existing interpreter path:

```sh
systemd-run --user --wait --pipe --collect \
  --unit=r0-comparison-probes-UNIQUE -p Delegate=yes \
  -p WorkingDirectory="$PWD" --setenv=PYTHONPATH="$PWD/src" \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  "$UV_PROJECT_ENVIRONMENT/bin/python" "$PWD/scripts/comparison_resource_probe.py" \
  --out /tmp/r0-comparison-resource-fresh.json \
  --walkthrough-out /tmp/r0-comparison-enforced-fresh
```

The probe manages only its dedicated service subtree and requires a fresh output.
It verifies child membership, a one-second deadline, a 64-MiB memory cap, and whole
cgroup termination with a spawned descendant. It then runs the three-task external
walkthrough under production limits (one CPU allowance, one numerical-library
thread, 8 GiB, no swap, 300 seconds per attempt). [Kernel receipts](resource-enforced.json)
record TIMEOUT, MEMORY_LIMIT and removed child groups. Production-cap fixtures
completed; this is not real-model feasibility evidence.

## Remaining external gates

Real TabPFN environment compatibility, model outputs and query invariance are
NOT_RUN. Real-data admission, revised #25 acceptance, protocol freeze, learner
selection, access and run budgets remain PENDING under their existing owners.
The all-missing numeric histogram-boosting incompatibility remains an explicit
failed attempt; the approved tree recipe was not silently changed. Training/eval
smoke commands depend on T7/T12 and are separate from these fixture checks.

## Final source verification and independent review

Ordinary `make check` passed: **202 tests**, strict Pyright with zero errors,
four import contracts, formatting/lint and index guards. No sandbox workaround
or diagnostic suppression was needed. The [completion receipt](completion-receipt.json)
binds the final module file hashes to all **45 successful cgroup-enforced fixture
attempts**. Raw bundles remain at the recorded local path; their integrity hashes
allow inspection independently of the summary. [Check log](completion-logs/check.log)
and [RED logs](completion-logs/) retain the actual before/after procedures.

Two independent reviewers inspected the working-tree delta atop `7146876`.
The first spec review rejected missing executed-script bindings; the standards
review found malformed external output could escape failed-attempt retention.
Both defects were reproduced, fixed and independently rechecked. The spec reviewer
passed five external-boundary tests; the standards reviewer passed eighteen focused
review/external/TabPFN tests and found no remaining hard standards violations.
These corrective verdicts do not erase the original rejection or establish real
model acceptance. Static inspection separately confirmed the constructor arguments
and preprocessing dictionaries against the pinned TabPFN wheel.

In required order after `make check`, `make smoke` and `make eval-smoke` both exited
2 because this branch has no `train` or `eval` command. Their logs are retained in
completion-logs. Smoke loss: **NOT_RUN**.

| eval-smoke task | metric | value | status |
| --- | --- | --- | --- |
| dev suite | unavailable | NOT_MEASURED | eval command not implemented |
