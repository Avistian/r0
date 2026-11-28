# Observed-task duplicate audit (M30-04)

`task_content_signature(task)` hashes the observed inputs, all labels,
context/query index arrays, missing mask, task type, class names and feature
schema. Task IDs, seeds, mechanism records, privileged raw observations and score
arrays do not enter the signature. The version is `observed-task-signature-v1`.

This is an exact observed-task comparison, not a mechanism equivalence test.
Repeated observations with distinct task IDs collide intentionally. Different
samples from the same function generally do not collide. Query labels enter this
offline diagnostic; this does not authorize exposing them to the learner or using
them for preprocessing. Row order, column order, schema, dtypes and split order
matter. Floats are not rounded; only signed zero and NaN payloads are normalized.
Tensors are serialized in contiguous little-endian form with length-delimited
metadata and bytes, then SHA256-hashed. Hash equality is the duplicate criterion,
not a proof against theoretical cryptographic collisions.

`index_coverage_shard(plan, index, output, database, summarize=True)` verifies the
shard using the existing archive/request checks, then loads completed tasks one
at a time and records signatures in a SQLite database. It locks the source shard
against writers during indexing. The database binds to the full plan identity
and signature version. One transaction indexes a shard: failures roll back that
shard's changes. Re-indexing unchanged tasks is idempotent. An existing ID with
changed contents, missing previously indexed task or incompatible index fails;
it does not silently rewrite the historical record.

The index is on disk and grows with committed IDs. Python holds one loaded task
and the shard inspection ID sets; SQLite uses an 8 MiB page-cache target and file
backed temporary storage. This is not a measured peak-memory guarantee for all
task shapes. Use local reliable storage for the SQLite database. Do not assume
SQLite locking is safe on an arbitrary mounted Drive filesystem. For Colab, keep
the live index on local runtime storage and reconstruct it from persistent
archives after a reset, or copy a closed database as an offline checkpoint.
Never copy an actively written database as a durable snapshot.

Use `summarize=False` for intermediate shards in a large audit: it reports only
that shard's indexed count and avoids a full aggregate scan after every shard.
Request the aggregate summary on the final indexed shard. Even the final summary
can take time and temporary disk space at 100M tasks. No 100M performance or index
size claim has been measured. Index disk admission limits remain caller-owned.

The summary reports total indexed tasks, distinct observed-task signatures,
duplicate excess (`tasks - unique`), excess fraction, number of duplicate groups,
largest duplicate group, rows and query targets. Per-stratum counts use all three
denominators. A duplicate group of size two contributes one duplicate excess,
not two. The report covers indexed committed snapshots only: unvisited shards,
rejected tasks and subsequently changed/deleted source files are not automatically
rechecked by looking at the index. Revisit every shard or build a fresh index with
writers stopped for final acceptance. Keep partial-audit scope explicit.

Near duplicates and mechanism equivalence remain `NOT_MEASURED`. These diagnostics
do not establish independent source-function coverage, useful learning diversity,
scientific quality or downstream benefit. Additional marginal/joint property,
correlation, severity and equal-budget comparison requirements remain open.

Focused checks include identical observations with different valid identity
records, changed labels and splits, single-representable-step input changes,
signed-zero normalization, idempotent re-indexing, changed-index rejection and
actual guarded generation across two shards. The initial duplicate fixture had
different context/query splits and was correctly counted as two tasks; it was
corrected to share the full observed-task contents. Raw evidence, including that
initial failure, stays in ignored `docs/evidence/issue-30-coverage/`.

A retained audit of the notebook's eight-task, four-stratum local pilot found
eight distinct observed-task signatures, 512 rows and 160 query targets. Each
stratum contributed two tasks, but polynomial tasks contributed half the rows
and 40% of query targets. This small example demonstrates why equal task quotas
do not imply equal row/query exposure. It is not a 100M diversity estimate.
