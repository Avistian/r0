# Issue 28: transactional raw matrix draws (candidate)

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
