# Generator device agreement

`scripts/check_device_agreement.py` compares the same recipe, seed and task ID
through the public generator on CPU and an explicitly selected candidate device.
Use `--candidate cuda` on the guarded Modal GPU lane. CUDA being unavailable is a
failure; the command never substitutes CPU. `--candidate cpu` is a local replay
preflight and provides no GPU evidence.

The default fixed plan contains 90 cases: ten families, three sizes and regression,
binary and multiclass targets. Families are linear, grouped linear, polynomial,
quadratic, random Fourier, periodic Fourier, radial, sinusoidal, exact GP and a
sum composition containing all eight analytic families. This tests representative
mechanisms; it does not qualify every sampled recipe or the whole sampling profile.

| Size | Rows | Features | Active features | Multiclass classes |
| --- | ---: | ---: | ---: | ---: |
| Small | 64 | 8 | 4 | 3 |
| Medium | 1,024 | 32 | 16 | 3 |
| Large | 20,000 | 256 | 256 | 100 |

Periodic Fourier uses at most 64 active features. Exact GP uses 64/512/1,024 rows
and 512 calibration rows. Other families use 1,024 calibration rows; composition
children use 256. Polynomial degree is four, quadratic rank 32, RFF terms 256,
periodic terms 64, and smooth terms `max(64, active features)`. Inputs are Gaussian,
regression noise is zero and class controls are neutral. Context contains three
quarters of the rows. These restrictions are part of the plan, not automatic
fallbacks when a case fails.

```bash
# Plan inspection generates no datasets and makes no CUDA availability claim.
python3 scripts/run_check.py python scripts/check_device_agreement.py \
  --candidate cuda --plan-only --out /tmp/device-plan

# Local CPU replay preflight; use a fresh output directory for every attempt.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_check.py python scripts/check_device_agreement.py \
  --candidate cpu --out /tmp/device-cpu
```

An optional `--plan` accepts a JSON object containing `source`, `request.seed`,
`request.start` and 1–128 uniquely named `agreement_cases`, each with `name` and
`recipe`. The source must match the current checkout, or `R0_GENERATION_SOURCE`
inside an image without Git. The output retains the exact plan bytes, their SHA256,
the driver SHA256, source revision, library versions, CPU thread count and actual
candidate GPU name. These establish provenance; they do not independently attest
an image's source environment.

For each case the command compares inputs, scores, targets, class probabilities,
context/query indices, source and episode identity, class vocabulary and mechanism
metadata. Numeric tolerance is `rtol=atol=1e-10`. Identity and metadata must match
exactly. CPU and candidate tensor devices are checked explicitly.

The first failed case stops execution and exits nonzero. Completed case records
remain, and the summary distinguishes passed, failed and not-run cases. An
unavailable GPU records `CUDA_UNAVAILABLE` with every case not run. A process killed
by an enclosing timeout may leave only started records and completed case files;
absence of a terminal summary is not a pass. Outputs refuse reuse or overwriting.

The command does not impose wall-time, address-space or provider spend limits.
The caller must enforce them. Peak RSS is cumulative for the Linux process;
CUDA peak allocation covers PyTorch allocations, not all provider GPU memory.
Per-case elapsed time includes both generations and comparison. No arrays are
saved by this command: retained recipes allow replay, while case records retain
identity and maximum observed absolute numeric differences. Passing this suite is
numerical agreement evidence, not training-quality, throughput or frontier-model
performance evidence. Wiring and actual GPU execution belong to the guarded Modal
workflow; neither plan-only nor CPU runs establish those requirements.
