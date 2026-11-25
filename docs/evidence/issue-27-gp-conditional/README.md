# Conditional GP inspection: CPU evidence

Source: `fd9e169fe87cc5389d1e5de2760844eafc7ca587`. The public API walkthrough
ran from that clean commit in `/tmp/r0-gp-conditional-fd9e169`. `files.json` records
the raw artifact hashes and sizes. The original summary is split losslessly into
`summary.json` (metadata), `cases-0.json` and `cases-14.json` (ordered cases) to
respect the repository's text-file limit. Selected results and all 1,024 draws
from the sampling check are retained here. Full per-case results remain in the
raw directory. This is local CPU API evidence, not browser or hosted evidence.

## What was checked

- All seven GP kernels, their sum and product, each for regression, binary and
  three-class tasks: 27 cases. Each reconstructs requested saved values with zero
  error, replays additional draws, and preserves the original task fingerprint.
- One maximum-shape case: 1,024 dataset rows, 512 calibration rows, 256 features,
  four active features, 100 classes and 256 query points. Conditional inspection
  with three additional draws took 1.609 s on the recorded CPU. This is one timing,
  not a throughput target or repeated benchmark. Whole-process peak RSS was
  621,342,720 bytes, including generation, serialization and prior cases.
- An independent dense Matérn-1/2 solve supplies the expected conditional mean and
  covariance for 1,024 additional draws across 128 explicit seeds. The largest
  mean error was 1.454 standard errors; covariance maximum error was 0.07027,
  against maximum expected variance 0.93693. The retained NPZ contains the actual
  samples and independent expected values, not only a PASS string.
- A smooth one-dimensional RBF case fails strict conditioning. Explicitly allowing
  truncation retains 11 of 48 directions and records a maximum normalized support
  projection error of 2.7814e-6. Its approximate result is retained separately.
  It is not presented as exact conditioning on all saved values.

Reproduce the public API walkthrough in a fresh directory:

```bash
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/gp_conditional_walkthrough.py \
  --out /tmp/r0-gp-conditional-fd9e169
```

`independent.py` retains the scientific reviewer's separate weighted Matérn-1/2
plus RBF calculation with three channels and 48 support points. It constructs
covariances independently and solves a dense system, then checks saved values,
replay, repeated coordinates and query reordering. Run it with the same environment
prefix and `python3 scripts/run_check.py python
docs/evidence/issue-27-gp-conditional/independent.py`; `independent.json` retains the
reviewer's output.

## Checks and retained limitations

Before implementation, the test failed because `condition_gp` was absent. Separate
red runs showed the missing draws option and missing explicit rank-truncation
option. The broader walkthrough found a multiclass broadcasting error in a finite
value check; the regression test now covers all three task types. An earlier
walkthrough also failed when its fingerprint attempted to serialize an Episode
directly; the procedure now converts that record to ordinary data.

At the recorded source, `make check` passed **455 tests, 1 CUDA skip**, in 101.22 s.
Then `make smoke` and `make eval-smoke` each exited 2 because the repository CLI
does not implement `train` or `eval`; there is no training loss or evaluation table.
Source review found no blocking standards or scientific/spec issues. Its optional
duplication observation concerns the versioned symmetric-root reconstruction also
used by generation and graph inspection; no unrelated generator refactor is made.

The explorer panel, actual hosted Colab verification, and the rest of #27 remain
pending. No cloud run was launched for this change, and these checks establish no
trained-model performance claim.
