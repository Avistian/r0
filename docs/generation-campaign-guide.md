# Generate, persist and resume a dataset campaign

Use [generation-campaign.ipynb](../notebooks/generation-campaign.ipynb) for headless
CPU generation. The notebook pins generator source to
`9d6c0e90597233a4ea63c796486ad1d318a1c515` and installs that checkout's `uv.lock`
with Python 3.12 in a separate environment. Data cells invoke that environment,
not the notebook kernel's installed ML packages. Setup downloads dependencies;
local validation of the notebook does not prove hosted installation works.

The current notebook demonstrates polynomial tasks, mixed numeric/category trees,
tree ensembles and ensembles with missingness. It is a working pilot template,
not a declaration that four strata provide sufficient coverage for 100M tasks.
The full #28–#30 generator portfolio remains in progress; the source pin will need
an explicit update when later generator features are accepted. Existing explorer
work remains separate in #126.

## First run

1. Open the notebook in Colab and run its pinned setup in a fresh runtime.
2. Mount Drive and choose a named campaign directory under `MyDrive`. Keep that
   exact path after reconnecting. `/content` outside the mounted Drive is temporary.
3. Run the plan cell. It freezes eight requested tasks across four strata, with
   two attempts per stratum and two tasks per shard. Inspect the declared family ×
   observation cells before generation.
4. Run the bounded generation cell. Each shard uses its own time/byte guard. The
   default invocation admits at most four shards, uses one worker and stops on a
   process error. A zero process return code is not proof every task succeeded:
   inspect the outcome counts and retained failure records.
5. Run verification/reload. Check committed, failed and pending counts; verify a
   saved archive and inspect its context/query partition.
6. Rerun generation with the same settings. Completed shards should report zero
   newly generated tasks. Then perform the reset drill below before scaling.

For an existing local environment, use the same notebook data cells with `REPO`
pointing at the pinned checkout, `PYTHON` at its environment's Python executable,
and `OUTPUT` at a durable local path. Skip the Colab-only setup/mount cells. Do not
silently mix a current checkout with an older installed package; the first data
cell prints the imported source location.

## Resume after Colab stops

The persisted plan and archive receipts are the checkpoint. There is no single
volatile in-memory counter that must survive. Each shard has a fixed global ID
interval. Within it, the runner verifies completed archives and skips their IDs,
retains failed attempts and regenerates tasks that never acquired a valid
completion receipt. Small archive groups reduce the amount of work lost between
checkpoints. Runtime interruption can repeat computation; it must not duplicate
committed IDs.

After a runtime reset:

1. Reopen the notebook and rerun pinned setup. The environment may need reinstalling.
2. Mount the same Drive account and select exactly the same `OUTPUT` directory.
3. Rerun the original plan cell. It checks identity instead of overwriting the plan.
4. Rerun the same shard window. Keep seeds, group size, limits, template configs
   and plan unchanged. Completed groups are verified before being skipped.
5. Inspect the aggregate report and saved IDs. Keep failure logs and incomplete
   archives; they explain what happened before the interruption.

For the persistence drill, record a completed archive's receipt SHA256 and IDs,
reset the runtime, remount Drive and verify the same archive again before resuming.
Then confirm unchanged committed counts and zero newly generated tasks for fully
completed shards. This is the hosted durability evidence; a local repeat run is
only local resume evidence. Drive mount operations and local `fsync` do not prove
remote durability on their own.

Use one notebook coordinator per campaign output. Drive is not a distributed
locking service. Do not start a second notebook while the first may still be
running. On local filesystems the writer lock prevents concurrent shard writers;
filesystem-specific locking and persistence require target-runtime verification.

`RETRY_FAILED=False` preserves failed IDs without resampling substitutes.
Set it to `True` only when intentionally retrying the same task IDs, for example
after investigating a transient resource failure. The report retains recovered
failure history. Changing resource caps or recipes changes the frozen request:
create a new plan/output rather than editing `run.json` or deleting receipts.
Repeated deterministic failure is not solved by repeatedly retrying the same draw.

## Reading the artifacts and reports

| Artifact | Meaning |
| --- | --- |
| `requested-plan.json` | Notebook's frozen requested allocation and templates. |
| `coverage-plan.json` | Manifest bound to the generated shard output. |
| `shards/00000000/run.json` | Immutable request for one shard. |
| `groups/.../start-*.json` | IDs admitted for an attempt, not proof they ran. |
| `groups/.../failures-*.jsonl` | Recorded generation/validation failures. |
| `groups/.../data-*.zip` plus matching `.json` | Archive plus completion receipt; both required. |
| `100m-demonstration-plan.json` | Planning artifact only; no 100M execution. |

A campaign report keeps unstarted shards in the requested denominator. Committed,
unresolved failed and pending IDs partition requested IDs. Failed-then-committed
IDs describe historical recovery and are also included in committed. Admission
events include retries; they are not a precise execution count if interruption
occurred between admission and evaluation. An archive without a completion
receipt remains incomplete even if it appears readable.

Inspection verifies archive integrity, membership, task IDs and request binding.
It rejects duplicate committed IDs across archives. It does not replay every
numeric mechanism or measure near-duplicate functions. Reports say
`mechanism_diversity: NOT_MEASURED` for that reason. A complete campaign means
all requested IDs have committed output, not that the dataset is scientifically
sufficient. Stop writers for a final report: campaign inspection takes one shard
lock at a time, not one atomic snapshot of the entire campaign.

`predictor.npz` contains context inputs/labels and query inputs. `truth.npz`
contains privileged information including query labels. Cards and generation
internals support inspection/replay. Do not feed privileged truth or generation
internals into the learner as predictor features. `load_group` is an inspection
API that exposes both sides; loading is not permission to collapse that boundary.

## Designing broad coverage for 100M tasks

Treat 100M as independently sampled source tasks, not rows or repeated episodes
from one frozen function. The planner currently requires per-task recipe sampling
or sampled tree/ensemble construction by default. Other paths need qualification
before contributing to an independent-mechanism count. Setting
`require_fresh=False` explicitly drops this requirement; it does not prove diversity.

Build an explicit allocation across the supported source-backed options and
extensions, using [the delivery scope](generator-delivery-scope.md) and option
progress documents. Preserve source/adaptation/extension labels. Include:

- Families and mechanisms: the supported functional, causal/network, fitted,
  tree/forest/ensemble and other source-derived variants; track unsupported paths.
- Regression, binary and multiclass classification, including class-count regimes.
- Rows, feature counts, context/query sizes, active signal fractions and nuisance
  dimensions, covering small, typical and bounded extreme shapes.
- Root distributions, dependencies, topology/depth/width, nonlinearities,
  coefficient scales and interaction orders where supported.
- Noise scales/laws, missingness regimes, typed encodings and observation effects.
- Selected joint cells such as family × noise, task type × shape and mechanism ×
  missingness. Marginal coverage alone can hide absent combinations.

A `CoverageStratum` contains one explicit request template, integer weight,
minimum allocation and tags. Minimums reserve rare regimes first; remaining mass
uses exact weighted largest-remainder allocation. Tags must match actual template
intent; the planner does not derive them from generated arrays. Use pilot output
measurements to check realized properties, not just tag totals. Do not claim
exhaustive Cartesian coverage. Hierarchical/adaptive mixture policies, expanded
presets and additional diversity diagnostics remain separate unfinished work.

Keep weak but valid draws visible. Investigate numerical invalidity separately
from low signal or difficult class balance. Report failures by stratum before
reallocating: automatically replacing hard draws with easy ones biases the mixture.
The planner counts requested attempts, not a guarantee of 100M accepted outputs.
If a campaign needs a further batch to reach an accepted-output target, preserve
the original history and define a new disjoint global ID range explicitly.

## Scaling and resource measurement

Run representative pilots for each material family/shape/regime, then measure
throughput, serialized bytes, peak process memory, failure rates and output
properties on the intended hardware and storage. The tiny notebook pilot is
insufficient to extrapolate every generator. Include startup and Drive write costs;
large fitted/causal mechanisms may behave very differently from small polynomials.

For a measured stratum, rough serialized storage is allocated tasks × measured
bytes per task. Rough runtime is allocated tasks ÷ measured committed tasks per
second, with explicit allowances for setup, failure/retry work, inspection and
storage overhead. Sum by stratum; do not apply one cheap-family rate to the whole
portfolio. These are estimates, not guarantees. Inspect free disk and runtime
limits before each bounded window. Memory cannot be inferred from archive size.

The plan supports 100M IDs without constructing a 100M-element schedule. Shard
size sets a request slice; group size sets archive checkpoint frequency. Choose
both from pilot measurements. The notebook's 100M demonstration keeps the tiny
pilot time/byte caps: large shards may require several resume invocations or may
hit the immutable storage cap. It is not a production-ready resource configuration.
Set adequate limits in a new production plan before launch. Byte/time limits apply
to each shard, not to the entire campaign, and repeated invocations extend total
runtime. There is no automatic campaign-wide spend/storage admission gate here.

Assign explicit disjoint shard windows when scaling. Resume the same window after
interruption; advance only after checking its counts. Archive verification reads
saved bytes, so repeated scans of a 100M campaign are not free. Verify bounded
windows during generation and reserve a full stopped-writer audit for acceptance.
Separate campaigns must coordinate global ID ranges themselves.

## Validation status

The five data cells passed locally on the existing Python 3.12.3 CPU environment:
eight tasks across four strata committed, and rerunning the window generated zero
additional tasks. A separate process was killed with SIGKILL after its first
two-task archive committed; resume generated only the two missing tasks and
preserved the original archive SHA256. These are local checks, not hosted proofs.

Local data-cell execution, resume checks and retained pilot artifacts are recorded
under ignored `docs/evidence/issue-30-colab-generation/`. Setup/download and Drive
mount cells are not executed by that local check. Hosted installation, actual
runtime-reset durability, full 100M execution and downstream model benefit remain
NOT_RUN or NOT_MEASURED until direct evidence exists. No full-scale or paid run is
launched by adding this notebook.
