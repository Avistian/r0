# Bounded workers and a killable generation process

`run_generation(..., workers=N, persist=callback)` keeps at most N task futures in
flight. Results are written in task-ID order. The callback runs after each observed
failure and completed group; a provider error stops further admission. Threads
already admitted may finish, but their uncommitted work is never counted complete.
Only one coordinator writes a run directory. Worker count may change on resume.

`run_guarded(request, out, workers=N)` executes that coordinator in a separate POSIX
process group. The deadline includes child startup. Timeout or notebook interruption
kills the process group; earlier completed output remains available for resume.
The child sets OMP/OpenBLAS/MKL thread counts to one. It does not enforce a local
RAM limit; the hosted platform must supply one. GPU execution remains NOT_RUN here.

`sync_path` enables `sync PATH` after failure/group writes. This is intended for a
[Modal Volume v2](https://modal.com/docs/guide/volumes), whose documentation defines
that command as a persistence operation. Local tests do not prove remote durability.
The mount must contain the output directory. The Modal wrapper must perform a
final provider commit even if its child times out.

Run format `generation-run-v1` buckets groups into directories containing at most
1,024 group directories, avoiding a flat directory with millions of entries. The
unversioned format from the preceding development slice is rejected rather than
silently regenerating its tasks under another layout. Use a new output directory.

## Reproduce

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/generation
```

`test_two_workers_run_concurrently_and_persist_each_group` uses a two-party barrier
to require actual concurrent execution. `test_worker_count_does_not_change_saved_tasks`
compares archive member bytes from one and four workers. The guarded-job tests
execute a real child, verify its completed output, and force a startup timeout.
These tests failed on missing worker/guarded interfaces before implementation.

Colab/Modal entrypoints and their guarded launch wiring follow in the next slice.
Neither target environment has been executed or accepted by this local evidence.

## Review repairs

Independent review rejected `baa7e91`: storage exhaustion lost an admitted worker's
failure; a provider `ValueError` could be mistaken for an empty archive; a deadline
could be hidden by a failed-task status. All three cases were reproduced with
failing public-interface regressions before repair. The writer now explicitly
closes/drains its bounded worker stream, recording already-started outcomes. Typed
archive-limit/empty exceptions preserve provider failures, and deadline exhaustion
takes precedence over a completed interval containing failed tasks.
