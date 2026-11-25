# Shared-input composition CPU evidence

Sampler source: `d885ea1052122f2d7590a8d0a63de4cbd1886f1e` (PR #96).
Walkthrough source: `844c3fce8fa09bfc8fe0449e72d5802670221688`, adding only the
independent evidence program. These are five single-task CPU runs and one intentional
failure, not full issue #27 acceptance.

| Case | Coverage |
| --- | --- |
| 0 | Sum of all eight analytic representatives: linear, grouped linear, polynomial, quadratic, RFF, periodic, additive sinusoidal, ridge radial |
| 1 | Binary subset product, uniform inputs, weak separation and class offset |
| 2 | Multiclass depth-8 product with repeated leaf, one active feature, bounded Student-t inputs |
| 3 | Sum of 64 independently sampled linear polynomial components |
| 4 | Quadratic + ridge radial, 20,000 rows, 256 active features, 100 classes, 16,384 shared calibration inputs |

Every successful request saved one source task and resumed with zero generation.
Child mechanisms and repeated expression references do not count as additional
independent datasets. Records retain complete requests, archive identities/sizes,
selected context-row arithmetic, results and restarts.

Case 4 recorded **488,763,392 bytes** child peak RSS and **1.353503076 seconds**
generation and persistence. Other cases used 254,709,760–255,868,928 bytes peak RSS.
Each request allowed one task, 120 seconds and 128 MiB output storage. This storage
limit is not a process-memory cap. Individual measurements are not stable throughput
estimates, a full Cartesian coverage result, or a 100-million-task qualification.

`failure.json` retains the extremely small component strength `5e-324`: nonzero
values would round to zero, so the run reports composition-product underflow. It
retains the request and failed task ID, with no replacement, retry or completion.

## Reproduce and inspect

Use a fresh output directory and an existing compatible environment:

```sh
git checkout 844c3fce8fa09bfc8fe0449e72d5802670221688
UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/composition_walkthrough.py \
  --out /tmp/r0-27-compositions-proof-reproduction > /tmp/r0-27-compositions-reproduction.json
```

Original archives remain at `/tmp/r0-27-compositions-proof/{0..4}/groups/`, with
failure events under `/tmp/r0-27-compositions-proof/failure/`. These temporary local
paths are not durable published artifacts. Requests, source and environment permit
regeneration; checksums bind this run, not all dependency/platform combinations.

The independent scalar procedure reads saved predictor inputs and mechanism cards.
It evaluates each leaf with Python scalar formulas (including a full pairwise
quadratic expansion), then evaluates the expression with `math.fsum`/`math.prod`.
It never calls the production signal evaluator. One saved context row and output 0
per case are compared to archived truth, using saved calibration statistics. Maximum
absolute discrepancy was **7.78e-16**, within relative/absolute tolerance 1e-9.
This is separate from archive checksum verification and broader mathematical tests.

The 64-component request is split into files to keep each text artifact reviewable.
Reassemble it without changing any field:

```python
import json
from pathlib import Path

root = Path("docs/evidence/issue-27-compositions")
record = json.loads((root / "cpu-3.json").read_text())
parts = json.loads((root / record["request_file"]).read_text())
request = parts["request_without_components"]
request["recipe"]["components"] = [
    component
    for filename in parts["component_files"]
    for component in json.loads((root / filename).read_text())
]
```

## Review and verification boundaries

Initial `d18bf45` was rejected on two scoped findings: child split/context settings
incorrectly changed parent source identity despite identical generated data, and
explicit all-zero child class offsets were rejected. Public regression cases retained
both failures. `d885ea1` preserves full replay recipes but excludes nested split/context
controls from parent source hashing and accepts neutral offsets by value. Independent
review passed all 11 composition tests and verified direct/nested child identity,
parent split behavior, replay and nonzero-offset rejection. The initial findings
remain valid historical evidence. Standards review found no violations; its optional
naming concern was resolved.

Author full check: **429 passed, one CUDA skip**. All 32 standalone compatibility
records still match the baseline in `docs/signal-evaluation.md`. `make smoke` and
`make eval-smoke` exit 2 because train/eval CLI commands are absent; no loss/table exists.
These author runs are distinct from independent correction and evidence reviews.

Environment: Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu, aarch64, one numerical
thread. Cards/receipts retain metadata. GPU, hosted Colab, explorer rendering,
exploration profiles and repeated resource measurements remain pending. No cloud
execution or provider billing occurred for this evidence.
