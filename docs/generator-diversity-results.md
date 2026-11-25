# Sampled CPU generation results

The predeclared [sampled matrices](generator-diversity-qualification.md) ran on
2026-11-25 at source `cf560e39c36bc884444617f8454d2ef7b2bf98bc`.
The broad suite finished with **260 PASS and 10 TIMEOUT**. All **108 adapted-reference
tasks passed**. There were no other failed, incomplete, unknown-launch or unlaunched
cases. The broad command returned 1 and the reference command returned 0, as intended.
No failed case was retried, resized, replaced or omitted.

Environment: aarch64 Linux, Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu, one numerical
thread. Each process had a 120-second wall deadline, an 8-GiB virtual-address-space
limit and 512-MiB output limit. These are local CPU results, not GPU/Colab acceptance.

## Broad profile: selected, saved and verified

Each family has 27 predeclared draws. PASS requires generation/save, exact replay and
selected-row graph reconstruction inside the worker deadline. The separate artifact
audit then verified each PASS task's selected recipe, archive, identity, shape and
finite values without generating new tasks. Verified source identities are unique
within each suite. This is an author-run artifact audit, not a new mathematical oracle.

Input counts below describe fully verified tasks. G = Gaussian, U = uniform,
T = Student-t and M = two-Gaussian mixture. A dash denotes an unsupported input
distribution for that family, not an unobserved supported setting.

| Family | Selected | Recorded saves | Fully verified | Timeout | G / U / T / M verified |
| --- | ---: | ---: | ---: | ---: | --- |
| Linear | 27 | 27 | 27 | 0 | 27 / – / – / – |
| Grouped linear | 27 | 27 | 27 | 0 | 6 / 10 / 7 / 4 |
| Polynomial | 27 | 27 | 27 | 0 | 9 / 5 / 9 / 4 |
| Quadratic | 27 | 27 | 27 | 0 | 3 / 13 / 8 / 3 |
| RFF | 27 | 27 | 27 | 0 | 27 / – / – / – |
| Periodic | 27 | 27 | 27 | 0 | 27 / – / – / – |
| Radial | 27 | 27 | 27 | 0 | 5 / 5 / 10 / 7 |
| Sinusoidal | 27 | 27 | 27 | 0 | 4 / 8 / 12 / 3 |
| Exact GP | 27 | 24 | 20 | 7 | 3 / 4 / 7 / 6 |
| Composition | 27 | 27 | 24 | 3 | 11 / 5 / 5 / 3 |
| **Total** | **270** | **267** | **260** | **10** | |

All timeouts occurred in the large shape. Seven timed-out cases had already
published successful generation/save measurements: four GPs and all three
compositions. Their recorded generation/save times were 54.36–115.57 seconds.
Verification did not finish inside the total deadline, so they remain TIMEOUT,
not PASS. Three other GP cases had no completed-save measurement; that absence
does not prove that no files were written. All original records remain retained.

Fully verified polynomial draws covered degrees 1–3 and 5–15; this random matrix
did not select degrees 4 or 16. Quadratic factor counts ranged from 10 to 254.
RFF draws covered every profile term count from 64 through 4,096. Compositions
included 1–58 components, every actual depth 0–8, and interaction orders up to 58;
these are observed extrema, not every intervening component count. All eight
analytic child families appeared in verified compositions. Verified GPs included
4–63 kernels, but those extrema do not qualify every shape at those kernel counts.
Earlier fixed/maximum-complexity examples remain separate evidence.

## Adapted reference and resource interpretation

Power, quadratic, RFF and product each passed all 27 draws. All inputs are standard
Gaussian. Observed power orders include 2, 3, 4 and 5; quadratic uses 20 factors,
RFF uses 256 terms, and product uses exactly two children at depth one. These are
r0 adaptations of selected published settings, not reproduction of the full prior.

All three repetitions passed in 85 of 90 broad family/shape/type groups and all
36 reference groups. Timing ranges and arithmetic 100-million-task projections
are retained per group. No projection is emitted for a group with a timeout.
Passing three draws is not a guarantee about rare failures or future latency.
Whole broad/reference profiles remain exploration distributions, not bulk-training
eligibility claims. Exact GP remains a reference generator. Model performance,
provider billing and 100-million-task execution were not measured here.

## Retention and reproduction

Raw manifests, requests, archives, process outcomes, timing reports and separate
`diversity-audit.json` reports remain under the ignored directories
`docs/evidence/issue-27-broad-qualification` and
`docs/evidence/issue-27-reference-qualification`. Reproduce using the two commands
in the protocol, then the artifact-audit command in the [coverage map](generator-coverage.md).
Use fresh output directories and the identified generation source; run the auditor
from `f1cc23cc34fea4b454a49ce36fd92069091c386d` or an identified compatible revision.
The auditor checks saved artifacts and deterministic selection, not independent formulas.

Report SHA-256 values:

- Broad: `4c6ecd00478974641c0b3e736977d5b37fe3c5c7351e9091b2ed2406a7bcb57f`.
- Reference: `03f9a6386d3d575bf35779ce924b2fc97d0d70106420a14c7049a8d821a78a44`.

The audit source and driver hash are embedded in each audit. Both artifact audits
passed with zero new generations; that verdict preserves the ten benchmark timeouts.
Full local checks at the generation source passed 483 tests with one CUDA skip.
