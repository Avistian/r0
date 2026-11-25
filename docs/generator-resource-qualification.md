# Repeated generator resource measurements

The `fixed-cpu-qualification-v1` suite measures fixed recipes through the public
saved-generation interface. It separates these timing workloads from the broad
[sampling profiles](sampling-profiles.md). A passing case qualifies only its
listed settings on the recorded local CPU environment. It does not establish
production mixture weights, general bulk eligibility, GPU throughput or model quality.

See the [measured fixed CPU results](generator-resource-results.md) for the retained
270-task run. CI exercises only the short preflight and interruption-reporting checks.

## Measurement matrix

The default matrix has ten families, three sizes, three task types and three
independent task identities per case: **90 cases, 270 measured tasks**. Every
repetition starts in a fresh process with root seed 27 and task identifier 0, 1 or 2.
There is no warmup or best-run selection. All requests are retained before launch.

| Size | Total rows | Features | Active features | Context fraction | Multiclass classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 64 | 8 | 4 | 3/4 | 3 |
| Medium | 1,024 | 32 | 16 | 3/4 | 3 |
| Large | 20,000 | 256 | 256 | 3/4 | 100 |

Exact GPs use 64, 512 and 1,024 task rows respectively, with 512 independent
calibration rows. Periodic tasks use at most 64 active features, including in the
256-column large case. These are named limits, not substituted generators.
Binary cases have two classes. Regression uses noise/signal ratio 0.5;
classification uses separation 1 and zero class offsets. Inputs are standard
Gaussian throughout this fixed suite.

| Family | Fixed settings |
| --- | --- |
| Linear | Independent normal coefficients; shape/sparsity above. |
| Grouped linear | Group size 4; 1,024 calibration rows. |
| Polynomial | Maximum degree 4; 1,024 calibration rows. |
| Quadratic | Rank 32; linear and interaction strengths 1; 1,024 calibration rows. |
| Random Fourier features | 256 frequencies; length scale 1; 1,024 calibration rows. |
| Periodic | 64 terms; length scale 1; 1,024 calibration rows. |
| Radial / sinusoidal | Additive projection; max(64, active count) terms; length scale and width spread 1; 1,024 calibration rows. |
| Exact GP | One RBF kernel; length scale 1; 512 calibration rows. |
| Composition | Linear + (degree-2 polynomial × 64-term radial ridge); 1,024 parent calibration rows. Child settings retain active dimensionality; their discarded setup observations use 16 rows. |

Maximum shape is different from maximum mechanism complexity. This suite does not
qualify degree 16, 4,096-term approximations, 64-component/depth-8 compositions,
64-kernel GPs, non-Gaussian inputs or their Cartesian product. Those remain separate
coverage/resource cases. Broad sampling's settings histograms and small examples
are not substitutes for these measurements.

## Process limits and recorded boundaries

Default limits per repetition are 120 seconds of process wall time, 8 GiB of
address space and 512 MiB of saved generation output. These can be specified on
the command line; the manifest retains the values. The parent kills the entire
worker process group at the wall deadline. The worker applies `RLIMIT_AS` before
importing NumPy/Torch. A harmless 128-MiB allocation probe and sleeping-worker
timeout probe must pass before the matrix starts.

Address space is virtual memory, not an RSS reservation or a whole-machine memory
cap. There is no CPU quota or exclusive core; NumPy/Torch thread counts are fixed
to one. Peak RSS is measured inside each process. Host CPU/memory information,
software versions and the applied limit are retained. Local memory limits do not
establish CUDA/Modal/Colab enforcement.

The generation timer covers request execution, task construction, serialization,
archive verification and persistence performed by `run_generation`. It excludes
interpreter/library startup and the subsequent independent replay/graph check.
`process_seconds` additionally includes startup and verification. Generation peak
RSS includes imported libraries. Both RSS readings are cumulative process
high-water marks taken before and after verification; the later reading does not
isolate verification memory.

After saving, each task is reloaded through `load_group`, replayed from its concrete
recipe, and compared exactly on inputs, scores, targets, probabilities and split
indices. Generation graphs reconstruct the first two and last rows with absolute
signal/target tolerance 1e-9. This proves replay and graph consistency; the separate
family walkthroughs provide worked mathematical checks.

## Failures and estimates

Every outcome remains in the summary: PASS, FAILED, TIMEOUT, INCOMPLETE,
LAUNCH_UNKNOWN or NOT_RUN. INCOMPLETE means a confirmed process launch has no
terminal result. LAUNCH_UNKNOWN means a launch was requested but no process-start
record or terminal result confirms it. NOT_RUN has no launch request. Planned and
confirmed launched counts are separate. Receipts are published atomically; a damaged
worker receipt is recorded without erasing the parent's observed timeout or exit. A
saved timing is published before verification, so a later verification failure
does not erase a successful generation measurement. Failed cases are not retried,
resized or replaced. The tool requires a fresh output directory.
After retaining the terminal report, generation mode exits nonzero if any selected
task failed or timed out. The read-only `--summarize` mode prints outcomes without
launching work; its successful exit only means the report was read successfully.

For each case, the report lists successful timing min/median/max and the number of
every outcome. It supplies 100-million-task time/storage extrapolations only when
all repetitions pass and at least three repetitions exist. These are multiplication
of the measured median by 100 million, not measured scale execution. Time excludes
startup, verification, retries and platform overhead. Storage extrapolates one-task
ZIP groups; larger groups have different overhead. No production mixture is selected,
and cost remains NOT_MEASURED. Three draws are a limited sample, not a latency-tail
or failure-rate guarantee.

## Reproduce and inspect

From an installed checkout, using the existing environment wrapper:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/qualify_generators.py \
  --out docs/evidence/issue-27-resource-matrix
```

Set `UV_PROJECT_ENVIRONMENT=/absolute/path/to/.venv` for deliberate environment
reuse. Source and all three measurement drivers must be committed before launch;
the manifest includes their identity and hashes. Run only this matrix when comparing
its timing repetitions; do not concurrently run tests or another benchmark.

For a short preflight, add `--families polynomial gp composition --sizes small
--task-types regression --repetitions 1`. This checks execution, not qualification.
To inspect a completed or interrupted run without generating anything:

```bash
python3 scripts/run_check.py python scripts/qualify_generators.py \
  --out docs/evidence/issue-27-resource-matrix --summarize
```

Raw requests, archives, stdout/stderr, measurements, cap probes and reports remain
under the ignored output directory. This document records the protocol; measured
results must be attached to the matching source and run before making claims.
