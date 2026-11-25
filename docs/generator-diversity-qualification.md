# Sampled generator qualification

This extends the [fixed CPU measurement protocol](generator-resource-qualification.md)
with predeclared draws from both versioned [sampling profiles](sampling-profiles.md).
It measures the recipes actually selected, including failures. It does not select
a production mixture or establish model performance.

## Declared matrix

Run each profile separately. `broad-v1` has ten families, three sizes, three task
types and three draws per combination: 270 tasks. `tabiclv2-adapted-v1` has four
families and the same control matrix: 108 tasks. Root seed is 127 and task IDs are
0, 1 and 2 within each family/size/type. Different recipes identify different source
tasks even when those integer IDs match. The complete matrix is selected and saved
before any generation starts; no settings are chosen according to generation success.

| Control | Small | Medium | Large |
| --- | ---: | ---: | ---: |
| Rows | 16 | 512 | 20,000 |
| Features | 1 | 32 | 256 |
| Active features | 1 | 8 | 256 |
| Context rows | 1 | 384 | 15,000 |
| Multiclass classes | 3 | 3 | 100 |
| Regression noise/signal standard-deviation ratio | 0 | 0.5 | 2 |
| Classification separation | 0.1 | 1 | 4 |

Exact GPs use 16, 512 and 1,024 rows, with one context row for small tasks and
three quarters otherwise. Periodic mechanisms use at most 64 active features.
Adapted-reference quadratic and product use at most 20 active features. These
are explicit profile exclusions, not fallback substitutions. Binary tasks have
two classes. Small/medium class offsets are zero. Large binary has offset 2;
large multiclass offsets are evenly spaced from -2 to 2 across all 100 classes.
The small context deliberately permits absent classes; validity does not require
all classes to appear. The matrix is a set of joint control strata, not a causal
experiment isolating any one difficulty control.

Every draw samples its own mechanism settings and input distribution from the
selected profile. The manifest retains the original controls/profile/identity in
`selection` and the exact concrete generation request in `request`. The bounded
worker consumes that concrete request, saves it, and checks exact replay and graph
reconstruction. The shared saved-run path's per-task profile selection is separately
verified by `scripts/sampled_run_walkthrough.py`.

## Limits and interpretation

The same default limits apply as in the fixed suite: 120 seconds of process wall
time, 8 GiB of virtual address space, 512 MiB of output, one numerical thread and
a fresh process per task. All attempts and failures remain visible. A numerical
or resource failure is not retried with easier settings. The command returns 1
if any selected task does not pass, after saving the complete terminal report.
This expected reporting behavior must not be mistaken for a lost run.

Three draws per family/size/type are a bounded exploration sample, not a reliable
estimate of rare failures or tail latency. The distributions include high-complexity
settings; this suite does not assume that every selected recipe is bulk eligible.
Successful-case timing ranges exclude failed tasks, whose counts and reasons must
be reviewed alongside them. Extrapolations are emitted only for groups where all
three draws passed, and describe only those sampled settings on this machine.
They are not throughput guarantees for the full profile or a selected training mix.

The fixed suite's maximum shapes, earlier one-off maximum-complexity examples and
this sampled suite establish different facts. None proves their full Cartesian
product. Distribution histograms must distinguish selected settings from tasks
successfully generated and verified. Standalone family evidence must remain
separate from component settings inside a composition.

## Reproduce

Use fresh output directories and committed source. Do not run another benchmark
or test suite concurrently with the measurement.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/qualify_generators.py \
  --suite broad-v1 --out docs/evidence/issue-27-broad-qualification

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/qualify_generators.py \
  --suite tabiclv2-adapted-v1 --out docs/evidence/issue-27-reference-qualification
```

Set `UV_PROJECT_ENVIRONMENT` to the deliberately reused environment if needed.
For a short preflight, add `--sizes small --task-types regression --repetitions 1`.
Raw evidence remains ignored and local. GPU/hosted acceptance requires its own
recorded execution; this protocol does not authorize cloud runs.
