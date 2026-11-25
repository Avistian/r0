# Stable calibration for polynomial and quadratic signals

Direct variance estimation squares centered values. A representable raw signal can
therefore produce zero or infinite intermediate variance: a standard deviation near
`1e-200` or `1e200` cannot be recovered by directly squaring its raw values in float64.
This is different from a genuinely zero signal or nonfinite raw function output.

Polynomial and quadratic generation now use the grouped-linear rescaling method
when the original variance estimate fails. Divide each output by its maximum
absolute calibration value, estimate mean and standard deviation in those units,
then restore the magnitude. The same calibration sample is reused; no RNG draw,
replacement task, size change, clipping or retry occurs. Nonfinite raw output and
zero or unrepresentable final scale still fail visibly.

One private helper shares this calculation. Grouped-linear uses it directly;
polynomial and quadratic preserve their previous arithmetic whenever that estimate
succeeds, retaining existing successful task records.

## Failure and correction

All eight new public cases failed before the change: polynomial degrees 1, 4 and 16
with input scales `10**(-200/degree)` and `10**(200/degree)`, plus pure quadratics
with input scales `1e-100` and `1e100`. They now match the corresponding unit-input-
scale normalized scores within `1e-10` absolute/relative tolerance. Full checks pass
309 tests with one CUDA skip. `make smoke` and `make eval-smoke` exit 2 because the
train/eval CLI commands are absent. These checks establish CPU behavior only.

## Successful-record compatibility

`scripts/calibration_compatibility.py` generates nine identified tasks: polynomial,
quadratic and grouped-linear, each with regression, binary and three-class targets.
Its digests include both identities, class vocabulary, complete recipe/mechanism
JSON, and each tensor's shape, dtype and bytes (including split indices).

The before/after JSON outputs were byte-identical. Each file's SHA-256 was:
`94674332eb1f7dcee13c6cdd999f9d6c9c4c9fc38717ee4444ff12380c4cb8cc`.
The baseline source was `e646fa8e814abeeb3568c8b4730672315762f1bb`.
This is a nine-case compatibility check, not all-configurations or GPU proof.

From the fixed checkout, reproduce using an existing environment and a new path:

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_repo=$(pwd)
git worktree add --detach /tmp/r0-calibration-baseline e646fa8e814abeeb3568c8b4730672315762f1bb
(cd /tmp/r0-calibration-baseline && python3 scripts/run_check.py python "$task_repo/scripts/calibration_compatibility.py" > /tmp/calibration-before.json)
python3 scripts/run_check.py python scripts/calibration_compatibility.py > /tmp/calibration-after.json
cmp /tmp/calibration-before.json /tmp/calibration-after.json
sha256sum /tmp/calibration-before.json /tmp/calibration-after.json
python3 scripts/run_check.py pytest tests/priors/test_polynomial_generation.py tests/priors/test_quadratic_generation.py tests/priors/test_grouped_linear.py
```

Existing raw resource measurements retain their original source revisions. The
normalization fix does not establish new throughput, GPU, hosted Colab or wider
#27 acceptance.
