# Exact coverage plans and resumable shards

`r0.modules.generation.CoveragePlan` allocates a finite campaign across named
`CoverageStratum` request templates. This is the first M30-01 allocation backend,
not completion of the mixture and diversity portfolio.

Each stratum receives its integer minimum first. The remaining attempts are
allocated proportionally to nonnegative integer weights using largest remainders;
equal remainders break ties by stratum key. A zero weight can reserve a rare-regime
minimum without taking residual mass. Minimums must fit the total. The total is
bounded at 100 million attempts. Failed attempts do not silently transfer quota
to an easier family or trigger replacements.

Templates have count 1 and start 0. `coverage_allocations` assigns disjoint global
ID intervals in key order, starting at the plan's start. `coverage_shard` produces
one bounded request slice without materializing the full task schedule. Changing
shard size changes plan identity; retain the same plan when resuming. Templates
may use different seeds, but global IDs never overlap inside a plan. Distinct
campaigns must coordinate ranges themselves if IDs must be globally unique
across campaigns.

The canonical JSON manifest includes all templates, weights, minimums, tags,
limits and shard controls. SHA256 identifies that manifest, independent of input
stratum/tag ordering. `coverage_plan_from_json` enforces the version and fields.
Plans have at most 4096 strata, 16 tag dimensions and an 8 MiB document limit.

`planned_coverage` reports allocations, marginal quotas and optionally selected
joint cells over 2–4 dimensions. Joint cells only include authored combinations;
nonzero marginals do not imply coverage of their Cartesian product. Tags are
caller-declared labels, not measurements of generated tasks. All strata must use
the same dimension names. Source/adaptation/extension labels can be included as
one dimension. This interface does not yet provide hierarchical allocation,
adaptive selection, curricula or measured output diversity.

By default every request must use per-task recipe sampling or a sampled
`tree-extension`/`tree-ensemble` construction. Other construction paths may reuse
one frozen mechanism; they are currently unqualified for this requirement.
Explicit `require_fresh=False` admits fixed or unqualified templates and records
that independent source sampling was not required. Unique task IDs and distinct
serialized recipes alone are not evidence of distinct mechanisms.

`run_coverage_shard(plan, index, output)` atomically publishes the manifest and
runs exactly one shard through the existing guarded coordinator. Reusing the
output with a different manifest fails. Resume verifies existing archives and
preserves failure history; failed-task retry remains an explicit argument.
The directory layout is `coverage-plan.json` and `shards/00000000/...`, with the
existing run headers, group archives and attempt records inside each shard.
POSIX locking is required. Filesystems without reliable `flock` need one external
coordinator per output/shard; a mounted drive is not automatically a distributed
lock or a proof of durable persistence.

Time and byte caps apply separately to each shard invocation, not the entire
campaign. No function here automatically launches 100M tasks. A 100M allocation
is planning evidence only. Campaign-wide admission, mechanism duplicate diagnostics and a representative
broad-coverage preset
remain outstanding; do not describe this arithmetic backend as a production
100M coverage guarantee.

## Validation

Focused checks exercise exact quotas with hand-calculated remainder allocation,
rare minimums, key-order invariance, global partitioning, JSON round trips and
invalid requests. A guarded three-task pilot saves two shards, verifies archives,
resumes without regenerating completed tasks and rejects a changed manifest.
A 100M plan checks allocation sum and random access to its final shard without
running generation. A joint-coverage fixture deliberately omits rare/low noise
and verifies that the report does not invent that cell.

Raw author evidence lives in ignored `docs/evidence/issue-30-coverage/`.
Full-suite acceptance of the current source and hosted Colab execution remain
separate requirements; focused checks do not establish them.

## Inspecting outcomes

`inspect_coverage_shard(plan, index, output)` verifies completed archive checksums,
archive membership, global task IDs and request binding. It rejects duplicate
committed IDs across archives, changed manifests/headers and active shard writers.
It does not regenerate data or perform numerical replay. Archives without their
completion receipt are counted as incomplete, never as saved datasets.

Reports distinguish unique admitted IDs, admission events (including retries),
recorded failure-log events, unresolved failed IDs, failed-then-committed IDs,
committed IDs and pending IDs. A start record proves admission only: a process
can die before evaluating any admitted task. Consequently these counts are not
an exact execution count or scientific rejection rate. Failure summaries supplement
historical failed IDs without inventing failure-log event counts. Successfully
recovered failures remain visible. Corrupt records fail inspection rather than
being silently discarded.

`inspect_coverage(plan, output)` streams existing shard reports into stratum totals
and committed tag marginals. Unstarted shards remain in the requested denominator.
Memory holds one shard's ID sets plus stratum totals, not all 100M task IDs. The
report locks each shard separately; stop all writers for a final campaign snapshot.
Tag marginals are declared-stratum accounting, not measured properties of saved
tasks. Exact-content/near-duplicate mechanism diagnostics and numerical diversity
remain NOT_MEASURED. `COMPLETE` means every requested ID has committed output;
it does not certify literature coverage, scientific quality or downstream benefit.

Outcome tests include a real guarded three-task run, historical recovered failure,
duplicate archive, unstarted shard and constructed interrupted-write records.
The interrupted fixture admits two IDs, records one failure and leaves an archive
without a receipt: the report must show zero committed, one unresolved failure,
two pending IDs and one admitted ID without an outcome.
