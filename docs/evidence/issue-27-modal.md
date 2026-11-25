# Guarded Modal generation setup

This slice adds a generation mode to the existing guarded workflow. It uses the
same request and worker as the local/Colab entrypoint; it does not enter training,
benchmark download, model evaluation or leaderboard recording.

The caller pins the matching factory implementation. Dispatch `agent-train.yml`
with `mode=generation`, a `configs/generation/*.toml` request, a run name and worker
count. Requests explicitly declare count, time and bytes; repository variables
cap all three. CUDA requests currently use one T4, requiring `MAX_GPU=T4`.
The adapter limits CPU to one physical core and memory to 4 GiB. Unsupported
requests fail before submission; no cheaper device or smaller task is substituted.

Admission requires `COMPUTE_ENABLED=true`, available credentials, an immutable clean
source revision, daily dispatch capacity and a budget reservation. All dispatches
that day count, including failures; workflow reruns are refused. Each run reserves
the configured `GENERATION_RUN_RESERVATION_USD`; the daily count times this reservation
must fit `GENERATION_DAILY_BUDGET_USD`. This is conservative admission accounting,
not a provider-enforced dollar cap or an actual billing measurement.

The resource estimate uses the published [Modal prices](https://modal.com/pricing):
one core, 4 GiB and optionally a T4 for generation plus a separate resume invocation.
Admission adds a 2x allowance and $0.25 for image building. Prices and actual build,
storage, network and infrastructure-retry charges may differ. Default workflow
submission time is 15 minutes; `GENERATION_JOB_TIMEOUT_MINUTES` may be set up to 75,
with generation/resume/startup allowances validated against it. Provider dashboard
budgets are independent; changing these variables does not authorize spending.

## Persistence and retry

The `r0-generation-v1` Volume uses version 2. Groups and failures are committed
through `sync /outputs`, and the adapter performs a final SDK commit. Each invocation
claims a distinct workflow-run/phase marker before generating. An infrastructure
retry returns a committed receipt or fails on an uncertain marker; it never silently
regenerates that invocation. A new dispatch is required to resume an uncertain run.

Two separate single-use containers perform generation and resume. The second must
verify all completed groups and report zero newly generated tasks. One coordinator
writes each run; workflow concurrency plus a single container bounds admission.
Do not run another independent app against the same run directory. Volume locking
is not a distributed writer lock. Failed tasks need an explicit retry flag, and
previous attempts remain retained.

For CUDA, the first invocation also compares CPU/GPU task identity, mechanism,
inputs, scores, targets, split and probabilities on regression, binary and three-class
64-row neutral recipes, with `rtol=atol=1e-10`. The admission plan validates these
separately from the requested recipe, preserving its classification controls.
This is a small agreement check, not coverage of all
shapes. CUDA allocation and child RSS are reported separately.

## Evidence and reproduction

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
python3 scripts/run_check.py pytest tests/generation/test_launch_caps.py
```

The local guard tests execute the public credential-free command, checking admission,
over-budget values, nonfinite reservation and rerun rejection. They do not call Modal.
The always-run workflow artifact contains the reservation, local CPU smoke and
available remote result. `actual_cost_usd` stays `NOT_MEASURED` until provider billing
is inspected. A missing remote receipt is an uncertain outcome, not proof of no work.

Hosted execution is initially **NOT_RUN**; attach exact workflow/source identities
and retained results after the authorized smoke. No 100M-task scale claim follows
from a small smoke. Colab target-runtime acceptance remains separate.

Independent review rejected the first adapter's diagnostic conversion: valid
classification offsets/separation became invalid regression controls after submission.
Credential-free CLI regressions now cover both requests and their separate valid
diagnostic recipes. The requested recipe remains unchanged. The factory pin update
also moves all callers together, as required by the existing workflow contract;
the other reusable workflow implementations are unchanged between those revisions.

Implementation follows Modal's [resource limits](https://modal.com/docs/guide/resources),
[Volumes](https://modal.com/docs/guide/volumes),
[infrastructure retry behavior](https://modal.com/docs/guide/retries), and
[serialized functions](https://modal.com/docs/sdk/py/latest/App).
