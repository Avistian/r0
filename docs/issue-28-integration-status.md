# Issue 28 integration status

The accepted scope remains [the full specification](specs/0028-causal-network-generators.md),
including every in-scope option in the recorded paper inventory. This checkpoint
is incomplete. Component reviews below do not establish workbench delivery,
scientific acceptance, or whole-issue completion.

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
| `60f13ba` | Converter input-channel allocation added; 17 focused tests and static checks passed. Evidence-location repair independently accepted. Channel-layout code has not yet received independent review or a full-suite result. |

Original rejection probes and failure logs are retained alongside repairs. Local
copies live under ignored `docs/evidence/issue-28-parallel/`, with a byte-verified
`manifest.json`. This directory contains both independent probes and separately
identified author/full-suite reports. It is local retention, not portable public
artifact delivery. The terminal `a22eb63` full-check log is included; later worker candidates are
not covered by that result.

## Parallel implementation ownership

The integration branch starts at `60f13ba` and includes the previously published
function/ancestor capture documentation. Workers start from the same source
checkpoint in separate worktrees:

- Geometry/archive: complete geometry sampling provenance, shared named context,
  mechanism construction, archive reopening, generation, inspection and pairs.
- Topology breadth: source-backed missing preferential attachment, motifs and
  acyclic rewrites, with explicit adaptation labels and raw replay records.
- Workbench: local and Colab workflows, construction controls, inspection/edit
  comparisons, save/reopen/export, and caller limits through generation transport.

Geometry owns priors construction APIs. Workbench owns explorer and minimal
`generation.RunRequest` limit propagation. Topology owns new sampler modules.
Changes are integrated only after reviewing their actual diffs and verification;
parallel authorship is not independent verification. Full-suite scheduling is
coordinated to avoid duplicate runs on the same candidate.

## Remaining delivery obligations

Geometry allocation currently feeds realized widths/readouts into an existing
request in a focused integration test. Its saved graph does not yet retain the
width/observation sampling records. This is the immediate geometry-track gap.

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
