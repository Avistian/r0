# Multiscale and spectral GP CPU evidence

Source: `cdbbba7ed1260c7a29a173665c7005f2f4e03618` (PR #91).
This is partial issue #27 evidence. Six single-task CPU runs succeeded and each
restart verified the saved group with zero new generation. Each JSON retains the
complete request, archive checksum and size, selected row, independently calculated
scalar, saved scalar, process result, restart result and covariance error bound.

| Case | Mechanism | Task |
| --- | --- | --- |
| 0 | Rational quadratic, alpha 0.1 | Regression |
| 1 | Rational quadratic, alpha 1 | Regression |
| 2 | Rational quadratic, alpha 100 | Regression |
| 3 | Eight weighted spectral bands | Binary |
| 4 | Subset rational-quadratic × spectral covariance | Multiclass, unbounded Student-t inputs with df 1 |
| 5 | Rational-quadratic + spectral covariance | 1,024 rows, 256 active features, 100 classes, 512 calibration inputs |

Case 5 recorded **480,182,272 bytes** child peak RSS and **21.011429978 seconds**
generation and persistence. The other cases recorded 256,659,456–259,051,520 bytes
peak RSS. Every request declared one task, 120 seconds and 128 MiB output storage;
the storage cap is not a process-memory cap. Each task count is one independent
source task. Observed single-task rates are not stable throughput estimates.

The intentionally unresolved near-constant RBF request is retained in `failure.json`:
one failed task, no completed task, no replacement or retry. This exercises the shared
numerical-failure path; the public tests separately cover spectral phase-limit rejection.

## Reproduce

Use an existing compatible environment, CPU and one thread per numerical library:

```sh
git checkout cdbbba7ed1260c7a29a173665c7005f2f4e03618
UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/exact_gp_walkthrough.py \
  --extended --out /tmp/r0-27-spectral-proof-reproduction > /tmp/r0-27-spectral-reproduction.json
```

Original raw archives remain at `/tmp/r0-27-spectral-proof/{0..5}/groups/`;
raw failure events are under `/tmp/r0-27-spectral-proof/failure/`. These temporary
paths are not durable published artifacts. Regeneration uses the retained source,
requests and environment. Archive bytes can differ across dependency/platform versions;
the receipts bind this run, not a universal cross-platform byte representation.

The retained program's `scalar_from_card` reconstructs covariance using the Gram
squared-distance identity and elementary rational-quadratic / product-of-cosines
formulas, then the symmetric square root. It does not import private sampler code.
For one saved context row and output 0 per case, it compares the scalar to persisted
truth with relative and absolute tolerance 1e-8. Maximum observed absolute difference
was **1.32e-13**. It uses saved normalization statistics; the public tests independently
recalculate calibration statistics for their smaller cases. Hash checking is separate
from this mathematical calculation.

Author verification: full `make check` passed (378 tests, one CUDA skip). Independent
spec review passed the 15 new focused tests and checked published formulas; standards
review found no violations. Neither review by itself establishes these resource figures.
`make smoke` and `make eval-smoke` exit 2 because train/eval CLI commands are absent.

The environment was Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu, aarch64. JSON
summaries retain actual metadata. GPU, hosted Colab, graphs, broad parameter-distribution
exploration, repeated resource measurements, 100-million-task extrapolation and full
issue #27 acceptance remain pending. Direct GP factorization is CPU work even when
task multiplication requests CUDA. No cloud run or provider billing occurred here.
