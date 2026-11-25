# Spatially varying GP CPU evidence

Source: `ff1737005114dcdb4f24d656137e4e6921073905` (PR #93).
Four single-task CPU requests succeeded, saved checksummed groups and resumed
with zero new generation. These records are partial issue #27 evidence.

| Case | Mechanism | Task |
| --- | --- | --- |
| 0 | Varying length, unit amplitude | Regression, 16 rows, one feature, 32 calibration inputs |
| 1 | Varying amplitude, constant lengths | Binary, 256 rows |
| 2 | Gibbs × spectral covariance on disjoint feature subsets | Multiclass, uniform inputs |
| 3 | Varying lengths and amplitude | 1,024 rows, 256 active features, 100 classes, 512 calibration inputs |

Each JSON retains the full request, archive receipt, selected context row, independent
scalar calculation, persisted scalar, run/restart result and covariance error bound.
The maximum absolute scalar discrepancy was **2.79625e-9**, below the declared relative
and absolute tolerance of 1e-8. The other cases differed by at most 3.56e-15. The small
one-dimensional case is more sensitive to covariance roundoff; it is not evidence of
bitwise agreement between differently evaluated matrix square roots.

Case 3 recorded **532,209,664 bytes** child peak RSS and **11.321485785 seconds**
generation and persistence. Other cases used 251,043,840–269,737,984 bytes. Requests
allowed one task, 120 seconds and 128 MiB output storage; output-storage limits do
not cap process memory. These are individual observed CPU runs, not repeated
throughput qualification or a 100-million-task capacity measurement.

`failure.json` retains a 32-component amplitude product whose final covariance
exceeds float64 range: one failed task, zero completed tasks, no replacement or retry.
The failure explicitly names combined kernel covariance. This differs from a
recoverable intermediate underflow where later amplitudes cancel earlier ones.

## Reproduce

Use the identified source and an existing compatible environment. Choose a fresh
output directory; numerical libraries used one thread each:

```sh
git checkout ff1737005114dcdb4f24d656137e4e6921073905
UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/exact_gp_walkthrough.py \
  --nonstationary --out /tmp/r0-27-nonstationary-proof-reproduction \
  > /tmp/r0-27-nonstationary-reproduction.json
```

Original archives remain at `/tmp/r0-27-nonstationary-proof-fixed/{0..3}/groups/`;
failure events are under `/tmp/r0-27-nonstationary-proof-fixed/failure/`. These are
temporary local paths, not durable published artifacts. The complete requests,
source, environment and program support regeneration. Receipts bind this execution;
archive bytes are not promised identical across platforms or dependency versions.

The retained `scalar_from_card` uses the elementary Gibbs formula with directly
squared lengths, envelope products, and a symmetric matrix square root. Production
uses scaled lengths and log covariance. This checks one saved context-row scalar,
output 0, per case against archived truth, using saved calibration statistics.
The public small-case tests separately recalculate those statistics. Archive hashes
and mathematical calculations are distinct checks.

## Historical rejection and correction

Independent spec review **rejected `34cbfb2`**: sequential covariance multiplication
could silently underflow before an opposing envelope restored the scale. A 64-component
product of 32 shrinking then 32 expanding envelopes produced normalized-score error
**1.6196457327** against an equivalent stationary kernel; alternating the same
components happened to pass. Earlier ordinary walkthroughs at that source had passed;
those passes did not establish the adversarial behavior was correct.

The public regression in `tests/priors/test_nonstationary_gp.py` retains that case:
uniform inputs at location 1.25, scale .25; 16 rows, one active feature, 32 calibration
inputs; seed 27 and task ID 5. Each Gibbs length is .001 and amplitude slope is -13
or +13. The opposite amplitudes cancel; the resulting reference RBF length is .001/8.

Correction `ff17370` accumulates signed log covariance before exponentiating for
products containing Gibbs kernels. Independent rerun measured maximum errors
**1.40e-13 grouped** and **1.33e-15 alternating**. A separate mixed spectral-product
identity preserves negative covariance. Spec review accepted the correction, passed
all 11 focused tests, and left the original rejection intact. Standards review found
no violations; its optional default-duplication concern was resolved.

Earlier raw output remains `/tmp/r0-27-nonstationary-cpu.json` and
`/tmp/r0-27-nonstationary-proof/`. These committed receipts use only corrected source.

Author full check: **390 passed, one CUDA skip**. Checkout, tracked-file, artifact,
format, lint, types, module boundaries and token guards passed. `make smoke` and
`make eval-smoke` exit 2 because train/eval CLI commands are absent; no training loss
or evaluation table exists. These suite counts are distinct from independent review.

Environment: Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu, aarch64. Cards retain execution
metadata. GPU/hosted Colab, explorer, broad exploration recipes, repeated resource
measurements and complete issue #27 acceptance remain pending. Raw GP draws are
Gaussian; sampled normalization changes their marginal distribution. Factorization
runs on CPU, including when later task multiplication requests CUDA. No cloud run
or provider billing occurred for this evidence.
