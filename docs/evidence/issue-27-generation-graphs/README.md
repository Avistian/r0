# Selected-row generation graph reconstruction

Source: `1405b9ed6ec95c08f59c0481e8c45a89a6e059da`. This is **local CPU graph-data
evidence**, not browser or hosted Colab evidence. Renderer work remains pending.

The walkthrough generated 35 tasks: all ten starting recipes in regression,
binary and multiclass modes; 4,096 Fourier terms; 64 composed linear mechanisms;
all seven GP kernels in both a sum and a product; and 20,000 rows × 256 features.
It reconstructed rows 0, 3 and the last row, expanded every composition child,
inspected late Fourier terms, checked graph parent identities, and fingerprinted
every task tensor plus provenance before and after inspection. Maximum root
signal error and target error were both **0.0** in these cases. This is not a
claim over the full parameter space or GPU execution.

`summary.json` retains source, case identities, seeds, task identifiers, data
fingerprints, node counts and measured reconstruction errors. Recipe construction
is retained in the source-pinned walkthrough. Raw full requests and all graphs
remain at `/tmp/r0-graph-walkthrough-final`; selected linear, multiclass-composition
and seven-kernel GP graphs are committed as `case-0.json`, `case-29.json` and
`case-32.json`. `files.json` records their hashes and the raw summary hash.

Reproduce with an existing environment and fresh output directory:

```bash
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/generation_graph_walkthrough.py \
  --out /tmp/r0-graph-walkthrough-final
```

The final shared-helper refactor preserves all 35 generated-task fingerprints
relative to `f2ff5865ef1043c7694321d4b7c051bf1674e530`. Run the same walkthrough at
that source with output `/tmp/r0-graph-walkthrough-f2ff586`, then compare:

```python
import json
from pathlib import Path

before = json.loads(Path('/tmp/r0-graph-walkthrough-f2ff586/summary.json').read_text())
after = json.loads(Path('/tmp/r0-graph-walkthrough-final/summary.json').read_text())
for left, right in zip(before['cases'], after['cases'], strict=True):
    for key in ('recipe', 'seed', 'task_id', 'identity', 'data_sha256'):
        assert left[key] == right[key], (left['name'], key)
print(len(after['cases']), 'task fingerprints unchanged')
```

The initial helper rename was separately checked against base
`5975c0f3af1d70a06031aa1a5b1bd4fb854f94cd`: `scripts/signal_compatibility.py` produced
identical `cases` arrays for its 32 standalone tasks before and after. Use the same
environment prefix as above, write each run's JSON to a separate file, and compare
`json.loads(path.read_text())['cases']`. The sorted-key JSON encoding of those
cases has SHA256 `de6481a240f4ea85592f1eb455111785c15884124366720ca59c548a121c4c81`.

Historical rejected checks remain visible: the first API test failed on the absent
entry point; initial reconstruction tests caught float32 calibration constants;
the first all-family walkthrough caught Fourier/smooth random-stream namespace
differences. Both were corrected before source `f2ff586`. Spec review then found
missing brackets in the displayed multivariate Gibbs formula, corrected in
`1405b9e`; standards review prompted shared mapping/reduction helpers. Both review
findings were closed independently. No generation recipe was changed to pass these
checks.

Final `make check`: **449 passed, one CUDA skip**; source unchanged since
`1405b9e`. The `train` and `eval` CLI commands remain absent, so `make smoke` and
`make eval-smoke` cannot produce training or evaluation evidence. No model-quality,
GPU or hosted-runtime acceptance is claimed here.
