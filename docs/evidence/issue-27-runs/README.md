# Resumable output groups: second code slice

`RunRequest` declares the recipe, seed, task interval, group size, device, admission
time budget and output-byte limit. `run_generation` writes groups and resumes the
same request. It skips only checksum-verified output from the same recipe/seed.
Changing the request requires a new directory. `attempted` counts this invocation;
`completed`, `failed` and `pending` describe the requested interval after it.

Groups contain lossless NPZ arrays and JSON cards. `predictor.npz` contains context
inputs/labels and query inputs; `predictor.json` declares classes and row positions.
`truth.npz` and `card.json` are privileged inspection files, not model inputs.
Completed archives have separate checksum receipts. Files are never overwritten;
attempt starts and results remain after retries. Incomplete archives are retained
without a completion receipt, and their tasks may be regenerated. An interrupted
attempt's actual work may be unknown; planned IDs are not claimed as completed.

The archive byte bound is checked before each member write, reserving ZIP headers.
The runner also reserves attempt metadata and counts existing files against its
total byte budget. It serializes one task at a time; it does not retain all task
arrays for the run. An exclusive local coordinator lock prevents concurrent writers.

## Reproduce

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/generation
python3 scripts/run_check.py python scripts/generation_walkthrough.py --out /tmp/r0-run-proof-fresh
```

Inspect `walkthrough.json`, the three task cards in the ZIP files, and
`groups/0/0/attempt-0.json`: the authored failure remains after successful retry.
The walkthrough requires a fresh directory to preserve prior evidence. Tests also
exercise process-like interruption, checksum corruption, wrong-recipe archives,
duplicate identities, missing completion receipts and storage exhaustion.

## Limits of this slice

Time is checked between tasks: it stops admission, not an in-progress computation.
Hard process timeouts, parallel workers, provider commit/flush, Colab/Modal wrappers
and GPU execution evidence are subsequent dependent slices of this delivery.
Local fsync/verification is not evidence of durable remote storage. Only one
coordinator may own a provider run directory. This PR does not complete #27 or
claim that 100 million tasks were generated. Initial public-interface tests failed
on the absent generation module/run API before implementation.

## Review repairs

Independent review rejected `dae7406`: a failure followed by interruption lost the
failure and permitted an implicit retry; 11 escaped 1,024-character errors wrote
67,517 bytes despite a 65,536-byte cap. The retained tests
`test_failure_is_retained_when_a_later_task_interrupts_the_group` and
`test_escaped_error_records_respect_total_storage_limit` reproduced three failures
before repair. Failures are now fsynced to an append-only log before another task
starts. Error text is bounded by its serialized JSON size, with space reserved for
both the log and attempt summary. These three regression cases pass after repair.
