# Issue 28 integration status

The accepted scope remains [the full specification](specs/0028-causal-network-generators.md),
including every in-scope option in the recorded paper inventory. This checkpoint
is incomplete. Scoped reviews below establish only the listed behavior. They do not establish
scientific acceptance or whole-issue completion.

## Retained construction checkpoints

| Source checkpoint | Result and limits |
| --- | --- |
| `665db66` | Explicit sampled graph candidate rejected for partial RNG publication with impersonating bit-generator subclasses. |
| `d56da97` | Concrete engine admission repair; scoped specification continuation passed. Full check: 1,705 passed, one skipped. |
| `6579ade` | Shared archive candidate rejected for source identity changing with episode partitions. |
| `d3f545b` | Source identity repaired; next review found missing caller-limit propagation through shared decoding and inspection. Full check: 1,740 passed, one skipped. |
| `ebc2cc0` | Shared exploration-limit propagation repaired; scoped archive review passed. No separate full-suite claim for this exact head. |
| `557bec8` | Cauchy topology sampler passed scoped independent specification and standards reviews. Full check: 1,749 passed, one skipped. |
| `7bbb608` | Complete topology/mechanism envelope passed scoped specification review. Full check: 1,755 passed, one skipped. Standards found a test importing a private decoder. |
| `ea25257` | Test now calls the public decoder; narrow standards repair accepted. |
| `a22eb63` | Latent-width and observation components passed 36 independent checks; standards found logs outside the required ignored evidence directory. Full check: 1,770 passed, one skipped. |
| `60f13ba` | Converter input-channel allocation added; 17 focused tests and static checks passed. Evidence-location repair independently accepted. |
| `756ec20` | Complete geometry archive independently checked across eight function families, 24 archive mutations, three resplits and six rollback cases. |
| `16bfd23` | Combined geometry and structural sampler checkpoint: full check 1,804 passed, one skipped, 661.80 seconds. |
| `103db8b` | Workbench review rejected parent-side pair reconstruction before the guarded worker and attempt receipt. Original failure evidence retained. |
| `3411bf9` | Pair reconstruction and comparison moved inside guarded worker. Independent backend and Chromium checks passed; failed timeout preserves active dataset and previous pair. Local construction/edit/reopen/export checked. Standards: no hard findings, P3 lifecycle duplication open. |
| `e668a1a` | Attachment/motif geometry integration independently checked: 684 budget configurations, 10 archive cases, 40 rollbacks; Cauchy archive unchanged byte-for-byte. |
| `b8d8234` | Combined workbench/topology successor, with legacy generated evidence removed from tracking. Full check running when this document was updated; no exact-head full-suite pass claimed yet. |

Original rejection probes and failure logs are retained alongside repairs. Local
copies live under ignored `docs/evidence/issue-28-parallel/`, with a byte-verified
`manifest.json`. This directory contains both independent probes and separately
identified author/full-suite reports. It is local retention, not portable public
artifact delivery. The terminal `a22eb63` full-check log is included; later worker candidates are
not covered by that result.

## Parallel implementation ownership

The current integration checkpoint is `b8d8234`, in
`/tmp/r0-28-current-integration`. It remains frozen during its full check.
Three separate implementation worktrees extend it or its documented ancestors:

- Native categorical variables: typed outcomes, one-hot descendant inputs,
  native targets, shared schemas and archive/inspection integration.
- Topology workbench: attachment/motif construction controls, save/edit/reopen
  and local-browser coverage; Colab shares the transport but hosted proof is separate.
- Sampled numerical pipelines: converter plans, independently fitted stage
  calibration, channel/node importance and structural-noise sampling with replay.

Parallel authorship is not independent verification. Each candidate needs its
own scoped review before final integration; one full check is scheduled for each
combined candidate. Generated logs, probes, receipts, archives and screenshots
stay under gitignored `docs/evidence/`. Authored implementation documentation and
reusable product verification scripts remain tracked.

## Remaining delivery obligations

Geometry archives now retain topology, width and observation sampling records,
shared mechanism decisions and all four stream boundaries. Cauchy, attachment
and motif/rewrite variants reach shared generation, inspection, resplitting and
supported task-control comparisons. This does not complete every topology or
observational profile in the accepted paper inventory.

The remaining scope also includes sampled converter/pipeline and structural-noise
options, native categorical variables and schemas, paper-specific mechanism and
network variants, matched/population hard and soft interventions, and all required
local/Colab editing and comparison workflows. Native categorical support cannot
be inferred from probabilistic label adapters. Task-level noise/separation pairs
cannot be counted as causal interventions.

Across-seed structural/functional diversity, filtering accounting, supported-size
and memory/throughput measurements, CPU/GPU agreement, actual browser/Colab
execution and inspectable portable evidence remain separate acceptance gates.
No paid or full-scale run is authorized by this design. Undisclosed source
algorithms remain explicitly unresolved; adaptations must not masquerade as exact
reproductions. No source option is closed by this status document.

## Bounded geometry census

At source checkpoint `a95422c` (source identical to `60f13ba`), 288 deterministic
CPU cases passed: 16 seeds for each combination of local/meta/global preference
mode, all-node/subset observation mode, and 2/8/32 nodes. Each case assigns 12
numerical input features and one target input channel, allocates converter spans,
and samples latent widths. An independent assignment count checks channel totals;
all cases and full records are retained, with no filtering or retries.

The retained procedure is
`docs/evidence/issue-28-parallel/geometry_census.py`; results, per-case hashes and
full records are in its `geometry-census/` directory. Reproduce from `/tmp/r0-28-parallel-integration`:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python docs/evidence/issue-28-parallel/geometry_census.py
```

Python 3.12.3 and NumPy 2.5.5 were observed. The complete component census
took 1.67 seconds on this run; this is not a full-generator throughput benchmark.
Peak memory was not measured. No categorical conversion, generated-table quality,
intervention, GPU, browser or Colab result follows from this census.


## Full-generation geometry census

At `b8d823412afc2fb7e56cc85ca332352048f4240c`, 36 attempted cases completed:
three topologies (Cauchy, attachment, motif/rewrite), three preference modes and
four seeds. Each uses nine nodes, 64 rows, four observed features, Gaussian roots,
linear functions, independent 16-row calibration and widths in [1, 4).
No attempt was filtered or retried. All recipe records, emitted arrays and
per-case receipts are retained locally in
`docs/evidence/issue-28-current-integration/full-geometry-census-v2/`.

Independent parent-order traversal measured four distinct topologies per family
across the four seeds (the three modes reuse each topology seed). Cauchy depths
were 0, 4, 5 and 8, retaining a valid completely disconnected draw; attachment and
motif depths were 3 and 4. All emitted values were finite; no observed feature had
zero sample variance. These are descriptive observations, not rejection rules.

Reproduce from `/tmp/r0-28-current-integration` using the retained program:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python \
  docs/evidence/issue-28-current-integration/full_geometry_census.py
```

The program intentionally refuses to overwrite its output directory; choose a new
output directory for another run. An initial harness setup omitted the required
Cauchy offset, failed before attempting any cases, and is retained separately.
Observed runtime was 6.94 seconds, Python 3.12.3/NumPy 2.5.5. Peak RSS was not
measured. This small Gaussian-linear census does not establish broad functional
diversity, filtering effectiveness, supported maxima, GPU or predictive quality.

Local browser evidence for repaired `3411bf9` is retained in
`/tmp/r0-28-workbench-repair/docs/evidence/issue-28-workbench-integration/`.
The independent pair probe forced a 0.001-second timeout, verified preserved
identities, reopened the successful pair and exported a CRC-valid ZIP containing
both datasets and the comparison card. Graph drill-down reconstructed repeated
rows with zero signal/target error. Mobile overflow and browser exceptions were
absent in the checked flow. Hosted Colab remains NOT_RUN.
