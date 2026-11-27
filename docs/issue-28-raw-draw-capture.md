# Issue 28: transactional raw matrix draws

This implements the draw-site recorder needed by the
[sampled graph design](research/0028-sampled-graph-ownership.md). It captures all
raw calls made by joint matrix, point and weight realization, including the Gaussian
values previously discarded when kernel matrices retained only their signs.
All five matrix families use it, including nested transformed/kernel constructions.
Complete sampled graphs and capture in function/root/prefix fitting remain open.

## Contract

Pass an optional `RawDrawLedger` to `materialize_joint_matrix`,
`materialize_joint_points` or `materialize_joint_weights` as `capture=ledger`.
Without capture, the existing returned realization schema and RNG draw sequence
remain unchanged. With capture, successful draws append immutable `RawDrawRecord`
entries. Each has the source plan's operation path, semantic role, stream label,
global sequence, distribution and parameters, shape, canonical dtype, actual raw
values, generator family and pre/post live-state digests. These plan paths identify
the plan being realized; they do not yet certify a full graph's typed ownership tree
or a distinct realization invocation. A full builder must bind record spans to its
construction receipts.

Recording happens at the draw site before sign conversion, scaling or matrix
normalization. The original raw Gaussian kernel values survive. Nested weight,
point, factor, assignment and jitter draws appear in their original order. Capture
includes raw numerical draws, not named hyperparameter decisions (those already
have a separate joint sampling ledger).

The recorder stages its records and the injected NumPy generator together. Success
commits both; any propagated failure discards both. Previously committed records
survive later failures. Captured draws must use the active transaction's bound RNG;
unrecorded advances are detected before a subsequent draw or final commit. A caller's
external RNG mutation is preserved and causes rejection of staged changes. These
checks are not a concurrency lock or a claim to undo arbitrary caller side effects.
One nested capture transaction binds one RNG; multi-stream graph-wide commit remains
an enclosing builder responsibility.

`RawDrawRecord.replay(rng)` requires the original live pre-state, re-executes the
recorded law, compares canonical bytes and final state, and commits only on success.
State digests do not recover a seed or prove authenticity. The implementation hashes
pickles of trusted live NumPy state only; it never stores or unpickles a caller's
pickle. Replay assumes the recorded Python/NumPy environment. Supported engines are
PCG64, PCG64DXSM, Philox, SFC64 and MT19937.

`RawDrawLimits` bounds draw count, retained raw numbers, estimated working bytes and
trusted state serialization size. Every draw checks its geometry and ledger charge
before allocating the draw array. A later cap can fail an already staged construction;
that attempt does not advance caller state. Graph-wide admission across plans,
realizations, ledger, fits and simultaneous arrays is still required. These estimates
are not measured RSS or OS memory limits. The record's array method assumes its
already bounded in-memory record; a public JSON decoder is not part of this slice.

## Checks and remaining work

The focused run passed 32 tests: all five matrix families preserve their existing
realizations and next-RNG values, every retained transcript replays the same random
stream, kernel normals remain lossless, nested capture rolls back after a late budget
failure, existing records survive later failure, all five engines replay, unbound
RNGs reject, unrecorded advances reject, and corrupted replay preserves caller state.
Several failing author tests exposed missing bound-RNG, replay rollback and
unrecorded-draw guards before repair. Their raw logs are retained under
`/tmp/r0-28-graph-raw-*.log`; intermediate source snapshots are not all retained.

Reproduce from the candidate worktree using the existing environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_raw_draw_capture.py \
  tests/priors/test_joint_matrix_plans.py
```

The entry point reached the repository file-size cap, so the existing fixture type and sampler moved verbatim to a private fixture file and remain re-exported publicly. Full checks and independent review are pending. This is not acceptance of full #28,
source-wide draw capture, complete graph provenance, browser/Colab/GPU behavior,
measured resources, scale or scientific performance. Next integrate capture with
remaining function/root materialization and fresh prefix populations, typed owner
validation, immutable graph construction receipts and final multi-stream commit.


## First independent rejection and repair

Candidate `ccaf90637bc19a8e1173cc5fc77946a206b5e223` received **SPEC REJECT**:
its general public recorder rejected legal noninteger or negative uniform upper
bounds because it reused discrete integer-bound validation. The matrix call sites
currently use other laws; no five-family materializer failure was established.
The reviewer stopped at this first defect, so broader independent comparisons were
not run. Immutable original evidence is retained at
`/tmp/r0-28-raw-matrix-spec-ccaf906`, pending publication.

The repair confines integer-bound checks to integers, permutations and choices.
Four literal uniform cases cover real bounds and negative upper bounds, checking
raw values, replay and next-RNG state against NumPy. The original failing author
run is `/tmp/r0-28-graph-raw-repair-uniform-red.log`; repaired execution uses the
focused command above. The repair still needs its own full check and independent
review. This does not erase the original rejected candidate.


## Published bounded verification

Main source `90de1cdcc26e8bc13cb40ac1808d7fe85f025308` has byte-identical
`src/` and `tests/` trees to repaired candidate `4100fe53`. Its full `make check`
passed **1278 tests, one skipped, in 524.54 seconds**. Original `ccaf906` also
passed its then-existing suite, but the independent uniform-bound REJECT still
stands; the earlier green suite did not establish that contract.

Independent [candidate review](research/0028-raw-matrix-capture-candidate-review.md)
and [exact-main review](research/0028-raw-matrix-capture-publication-review.md)
accept this bounded component: 255 baseline compatibility cases, 210 transcripts,
1655 records and 19240 raw numbers; seven laws across five engines; literal
matrix/point equations; nested failures; replay tampering and record budgets.
Standards found no documented violations; the original low-severity duplicate
engine-list heuristic remains. The fixture type and sampler retain identical ASTs
and outputs after their private-module move.

[Original REJECT](research/0028-raw-matrix-capture-original-reject.md),
[full check](research/0028-raw-matrix-capture-check-4100fe5.txt),
[standards](research/0028-raw-matrix-capture-standards-original.txt),
[repair delta](research/0028-raw-matrix-capture-standards-repaired.txt) and
[author logs](research/0028-raw-matrix-capture-development.json) remain separate.
The archive retains immutable verifier programs verbatim, pinned source archives,
commands, outputs, failures, environment and checksums. No formatted public program
copy is substituted for the executed verifier.

Recover and replay using the already installed compatible environment:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-raw-matrix-capture-spec-artifacts.json \
  /tmp/r0-raw-matrix-evidence --repo .
PYTHONDONTWRITEBYTECODE=1 python3 \
  /tmp/r0-raw-matrix-evidence/publication-independent/replay.py
```

The recovery verifies all six archive parts and all 96 retained manifest entries.
The portable replay extracts the pinned source archives into a fresh directory and
compares exact outputs/transcripts. Python 3.12.3, NumPy 2.5.5 and Torch 2.15.0+cpu
were used; dependencies are not installed by this command. Smoke/eval-smoke are
unavailable until the train/eval CLI exists. Full #28 remains unfinished.
