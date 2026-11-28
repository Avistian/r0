# Per-task structural construction

`generation.per_task_construction(source)` opts a `GraphConstruction` into a
new mechanism draw for each global task ID. Fixed constructions retain their
existing behavior. Supported source topologies are `explicit`, `cauchy`,
`geometry`, `attachment-geometry`, `motif-geometry`, `network`, and
`network-hyperprior`. Authored networks and unqualified constructors are rejected.

The construction seed and run seed select a deterministic odd affine permutation
of the 63-bit task-ID domain. Each ID therefore receives a distinct construction
seed within that pair; scheduling order and shard boundaries do not affect it.
Distinct seeds do not guarantee unique functions or datasets. Degenerate priors
can still repeat outcomes, so realized content audits remain necessary.

Saved tasks retain the full realized recipe and a `construction_sampling` receipt
with the version, original construction seed, run seed, task ID, derived seed,
and canonical source-control SHA256. Resume verifies this receipt against the
request as well as verifying the reconstructed recipe. Fixed requests reject
unexpected sampling receipts. Existing observation transforms compose with this
source sampling before archival.

```python
from dataclasses import replace

from r0.modules import generation as g

# source is an existing sampled GraphConstruction; request is its RunRequest.
fresh = replace(request, construction=g.per_task_construction(source), count=1)
plan = g.CoveragePlan(
    (g.CoverageStratum("structural", fresh),),
    total=100_000_000,
    shard_size=1000,
)
# Run only the explicitly selected bounded shard, then rerun it to resume.
result = g.run_coverage_shard(plan, 0, output_directory)
```

This single-stratum illustration is not the broad production campaign preset.
The seven constructor paths have focused checks for changed responses under
common episode randomness, exact saved-recipe replay, and fixed-mode stability.
A guarded network shard also exercises missingness, persistence and zero-work
resume with unchanged archive bytes. These checks do not measure full-campaign
throughput, mechanism uniqueness, or paper-option coverage.
