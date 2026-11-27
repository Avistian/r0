# Issue 28: function and root raw capture

This extends the matrix raw ledger through all eight source function families and
all four root base families. It is part of the complete sampled graph work in
[the accepted specification](specs/0028-causal-network-generators.md), not completion
of #28. The earlier matrix uniform-bound REJECT remains retained separately.

`materialize_joint_function` and `materialize_joint_root` accept optional
`capture=RawDrawLedger()`. Returned realization records and their numerical
recipes are unchanged. Capture records original values at their owning draw
sites, before transformations. Record order is global across the named streams;
`raw`, `fit` and `calibration` identify their respective actual generators.

Inventory:

- Linear/quadratic: nested matrices and the complete quadratic permutation above
  input width 20.
- Fourier: frequency uniforms, radial directions/factor/weights when selected,
  phase uniforms and Gaussian output coefficients.
- Tree: raw feature choices with probabilities, fit threshold-row indices and
  Gaussian leaf values.
- Discretization: full fit-row permutation and nested output matrix.
- EM: fit anchor rows, Gaussian jitter/spreads and nested output matrix.
- Product: both cheap children in order, including their separate fit choices.
- MLP: every ordered matrix and each activation's fit shift-row choices. Captured
  reconstruction uses the same numerical fitting helper after whole-MLP admission.
- Roots: covariance factor/weights, base values and ball radii, then the selected
  function's complete construction and fitting draws.

`RawDrawLedger.transactions` stages 1–64 distinct named NumPy generators and the
ledger together. Nested single-stream transactions inherit their existing name.
Nested multi-stream transactions must retain the same name-to-generator bindings.
The caller and yielded stream mappings cannot add, remove or replace bindings.
The five supported NumPy engines remain PCG64, PCG64DXSM, Philox, SFC64 and MT19937.

On a late construction/validation/record-budget failure, all construction, fit and
calibration draws and new records are discarded; earlier committed records remain.
Unrecorded RNG advances and false stream labels are rejected. External caller RNG
mutation is preserved and causes rejection; transactions require exclusive caller
ownership, not concurrent mutation support. State-byte admission is aggregate over
participating generators. Captured records remain replayable from independently
supplied trusted generators; no unpickling or public raw JSON decoder is added.

Author evidence includes genuine failing/passing checks for missing multi-stream
support and mapping mutation, plus all 32 root/function combinations across five
engines. Each combination checks captured/uncaptured realization equality,
record-by-record replay and next RNG values, then deliberately fails the last draw
budget and checks all streams and an earlier ledger record remain unchanged.
Existing literal-equation and realization tests also run. Full `make check` and
independent review are pending at source commit time; later evidence must state
its exact source identity. Smoke/eval-smoke remain unavailable without train/eval.

Still unfinished: prefix draws, complete sampled graph builder and typed
plan→realization→runtime ownership, independent ancestor fit populations, aggregate
whole-graph admission, raw decoding, broader paper options and end-to-end
local/Colab/browser/GPU/resource/scientific verification. This component does not
establish those requirements.

## Integration evidence and reproduction

Main function source `b8eec282efb22cb3a5bbe426e3c5feb1bf70a6e3` has the same
product `src/` tree as `a435427`, whose full `make check` passed **1455 tests,
one skipped, in 523.68 seconds**. Main source and tests match repaired `6a31732`.
That repair changes only tests: it removes a private-helper import and six internal
uncaptured-helper cases, retaining eleven public transaction tests. Those eleven
passed separately. The 1455-test run was not repeated at the test-only repair head;
independent verification separately retains both captured and uncaptured modes.

[Candidate review](research/0028-function-raw-capture-candidate-review.md) and
[exact-main review](research/0028-function-raw-capture-publication-review.md)
cover 280 cases and 1615 original NumPy calls, original/captured/uncaptured output
and next-RNG equality, both covariance bases, both RFF branches, quadratic width23,
every cheap product family in both positions, every MLP activation placement and
singleton discretization. All 160 core root cases also passed late-budget rollback.
The independent adversarial checks cover late validation, bindings, aliases,
external mutation, aggregate state limits and 0/1/64/65 stream boundaries.

[Full check](research/0028-function-raw-capture-check-a435427.txt),
[original Standards finding](research/0028-function-raw-capture-standards-original.txt),
[repair review](research/0028-function-raw-capture-standards-repaired.txt) and
[author raw logs](research/0028-function-raw-capture-development.json) retain their
separate identities. The low-severity duplicate root-draw recipe heuristic remains.
The original oracle's class-name-mangling harness error remains in its sealed packet.

Recover exact archived inputs and executed verifier programs, then replay into a
new directory using an existing Python 3.12.3 / NumPy 2.5.5 / Torch 2.15.0+cpu environment:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-function-raw-capture-spec-artifacts.json \
  /tmp/r0-function-capture-evidence --repo .
PYTHONDONTWRITEBYTECODE=1 /home/avist/Projects/r0/.venv/bin/python \
  /tmp/r0-function-capture-evidence/publication-independent/replay.py \
  --output /tmp/r0-function-capture-replay
```

The portable procedure extracts both source archives, executes ten source-pinned
steps and compares original-base, plain and captured outcomes; no original checkout,
dependency installation or network access is required. Its original successful
execution receipts and results are retained under `publication-independent/replay-results`.
No formatted program copy replaces the executed programs.

Source archives are reconstructed byte-for-byte from pinned Git commits before
manifest verification. Original candidate history is published on branch
`evidence/issue-28-raw-capture`; fetch that branch in a single-branch clone before
recovery. Identical retained files are restored from a checked duplicate-path map.
No raw evidence is dropped or normalized by this packaging.
