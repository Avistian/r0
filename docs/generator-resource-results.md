# Fixed CPU generator measurements

Measured on 2026-11-25 at source
`3dd2d3b42ee9d3012e6dafa1d0c3a78764efc364` using the
[fixed workload protocol](generator-resource-qualification.md). All **270 tasks
across 90 cases passed**, with zero failures, timeouts, incomplete or unlaunched
cases. Each case has three independent task identities. This establishes the
listed local CPU results, not general bulk eligibility or GPU/hosted acceptance.

Environment: aarch64 Linux, Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu,
one numerical thread, no exclusive core or CPU quota. Each process had an 8-GiB
virtual-address-space limit, 120-second wall deadline and 512-MiB output limit.
The memory and timeout probes passed before the matrix. Peak generation RSS
across the case summaries ranged from **238.7 to 465.3 MiB**, including imports.

## Saved-generation time

Seconds are medians of three fresh-process draws, measured from generation start
through verified archive persistence. Startup and later replay/graph checking are
excluded. Raw reports retain min/median/max, process time and cumulative RSS.
Small/medium/large shapes and fixed settings are defined in the protocol.
Large analytic cases have 20,000 rows and 256 features; exact GP has 1,024 rows.
Periodic large cases use 64 active features; other large cases use 256.
Multiclass has three classes for small/medium cases and 100 for large cases.

| Family | Task | Small seconds | Medium seconds | Large seconds | Large peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| linear | regression | 0.0486 | 0.0415 | 0.5376 | 408.6 |
| linear | binary | 0.0371 | 0.0396 | 0.5334 | 412.6 |
| linear | multiclass | 0.0424 | 0.0384 | 0.6883 | 465.3 |
| grouped | regression | 0.0429 | 0.0521 | 0.4562 | 372.0 |
| grouped | binary | 0.0521 | 0.0501 | 0.4333 | 376.0 |
| grouped | multiclass | 0.0399 | 0.0539 | 0.7534 | 439.1 |
| polynomial | regression | 0.0534 | 0.0712 | 0.6183 | 415.2 |
| polynomial | binary | 0.0540 | 0.0691 | 0.5160 | 418.9 |
| polynomial | multiclass | 0.0393 | 0.0603 | 0.9525 | 446.4 |
| quadratic | regression | 0.0668 | 0.0566 | 0.5164 | 385.5 |
| quadratic | binary | 0.0955 | 0.0784 | 0.4786 | 387.5 |
| quadratic | multiclass | 0.0548 | 0.0803 | 1.0206 | 438.1 |
| rff | regression | 0.0547 | 0.0659 | 0.7666 | 380.9 |
| rff | binary | 0.0668 | 0.0788 | 0.8052 | 382.5 |
| rff | multiclass | 0.0548 | 0.0876 | 1.0794 | 428.4 |
| periodic | regression | 0.0522 | 0.0595 | 0.6900 | 375.4 |
| periodic | binary | 0.0502 | 0.0638 | 0.6187 | 379.0 |
| periodic | multiclass | 0.0432 | 0.0737 | 0.8937 | 420.4 |
| radial | regression | 0.0475 | 0.0408 | 0.6246 | 380.8 |
| radial | binary | 0.0468 | 0.0428 | 0.7032 | 382.6 |
| radial | multiclass | 0.0484 | 0.0454 | 0.8939 | 424.9 |
| sinusoidal | regression | 0.0376 | 0.0499 | 0.7250 | 380.7 |
| sinusoidal | binary | 0.0418 | 0.0657 | 0.6394 | 382.4 |
| sinusoidal | multiclass | 0.0442 | 0.0470 | 0.8692 | 424.6 |
| gp | regression | 0.1534 | 0.6529 | 5.9721 | 402.3 |
| gp | binary | 0.2235 | 0.6842 | 5.7264 | 402.2 |
| gp | multiclass | 0.1528 | 0.6532 | 5.9675 | 402.2 |
| composition | regression | 0.0496 | 0.0745 | 0.9780 | 382.2 |
| composition | binary | 0.0518 | 0.0695 | 0.9192 | 383.6 |
| composition | multiclass | 0.0606 | 0.0660 | 1.7186 | 422.0 |

## What the scale estimates mean

The report includes per-case 100-million-task extrapolations. At the largest
100-class analytic shapes above, median generation/save time implies about
**797–1,989 single-worker days**, excluding startup, retries and platform overhead.
The GP reference at its smaller row limit implies about **6,907 days**. These
are arithmetic projections of this machine and these settings, not measured
100-million-task execution or estimates for a chosen training mixture.

One-task archive sizes also extrapolate to petabyte-scale storage for those large
cases. Grouping, streaming, task-size distributions and a production mixture change
the practical resource requirement; their performance is not inferred here.
No provider cost estimate or additional cloud spend was produced by this run.

## Verification and reproduction

The worker reloaded every saved task, compared replay arrays exactly, checked
finite inputs/scores/targets, and reconstructed first/second/last-row graphs.
A separate author-run artifact checker then verified **270 distinct source tasks**,
all archive checksums, requested array shapes, task identities, recorded address-space/
thread settings, timeout outcomes and all 90 summary calculations, with **zero
generation calls**. This is distinct from the
earlier independent scalar family walkthroughs and from independent code review.

Reproduce the matrix using the protocol command and a fresh output directory, then:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/verify_qualification.py \
  docs/evidence/issue-27-resource-matrix
```

Retained local files include `manifest.json`, `report.json`,
`independent-audit.json`, all requests, saved groups and process outcomes.
Report SHA-256:
`5988a2d562d3dce310e0394758189de5193b8de68e137f8cdf4f067ac4854cfe`.
The artifact checker hash is retained in its audit. Raw evidence remains ignored,
not committed. The CLI can summarize an interrupted matrix without launching work.

This fixed Gaussian-input suite does not qualify the complete broad-profile
distribution or every maximum-complexity combination. Existing family walkthroughs
supply separate one-off input/complexity cases; repeated broad-profile coverage,
the final eligibility matrix and target-runtime verification remain explicit
remaining #27 work. Exact GPs remain reference workloads.
