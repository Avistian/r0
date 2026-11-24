# Historical implementation and review evidence

The sections below retain the initial implementation/review state. For the PR
completion and current reproduction commands, read [completion.md](completion.md).
The original receipts remain unchanged; later fixes do not erase earlier failures.

# Offline comparison implementation and fixture review

This is fixture evidence, not protocol acceptance or real TabPFN evidence. The
accepted ticket supplied the design; `docs/specs/0026-offline-comparison.md` was
not present in this checkout. No data, model or dependency was downloaded.

## Reproduce

```sh
export UV_CACHE_DIR=/tmp/r0-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py python scripts/comparison_walkthrough.py \
  --out /tmp/r0-comparison-review
```

Use a fresh output directory. Each task produces one preparation bundle and 15
attempt bundles (three draws × five predictors). `learner_fixture` and
`tabpfn_fixture` are explicitly authored mean/frequency substitutes. They are not
model implementations or evidence about model quality. All comparisons remain
INCOMPLETE, with actual TabPFN UNAVAILABLE and actual learner quality NOT_MEASURED.

Separate commands use `python3 scripts/run_check.py python -m r0.modules.comparison`:

```text
prepare --source docs/evidence/issue-26/fixtures/binary.json --out /tmp/comparison-input
run --prepared /tmp/comparison-input --out /tmp/comparison-output \
    --fixture-diagnostic --fixture-predictors
tabpfn --prepared /tmp/comparison-input --out /tmp/tabpfn-fixture \
       --draw 1 --fixture-diagnostic
tabpfn --out /tmp/tabpfn-unavailable.json
```

Without `--fixture-diagnostic`, attempts require `--cgroup-root` pointing to an
owner-delegated Linux cgroup v2 hierarchy. Failure to enforce caps is a retained
FAILED attempt; there is no automatic fallback. The fixture diagnostic is not a
protocol attempt. The standalone worker receives input JSON and an output path;
scoring labels are copied into the attempt only after the worker has finished.
Separate environments can consume the same version-1 input/output contract.
Nothing installs TabPFN or changes either environment's dependency requirements.

## Follow a query

For binary draw 1, query ID
`e695a2cc9e0f7b5e46d7d7c93a8b9d25cc13ef1dbf32364785a27ae5448bc91d`
is first in the manifest, scoring JSON and every predictor input and output.
Declared class order is `[yes, no]`; the target is `yes`.

| Predictor | Saved raw probability vector |
| --- | --- |
| simple | `[0.5, 0.5]` |
| linear | `[0.43946838019208223, 0.5605316198079178]` |
| boosted_tree | `[0.5, 0.5]` |
| learner_fixture (substitute) | `[0.5, 0.5]` |
| tabpfn_fixture (substitute) | `[0.5, 0.5]` |

The simple predictor's eight context targets are
`[no, yes, yes, no, no, yes, no, yes]`. Query targets are
`[yes, yes, no, no]`. Each query contributes `(0.5-1)^2 + (0.5-0)^2 = 0.5`;
Brier is `(0.5+0.5+0.5+0.5)/4 = 0.5`. Declared-order argmax ties predict
`yes`, yielding accuracy `2/4 = 0.5`. Both binary probability columns count.

For regression draw 1, query ID
`4a200c519451e035b8466f79bcc2b5366923cf4e3c55dc8130dffdb56962be9e`
is first in all five outputs. Context targets are
`[31, 51, 39, 11, 53, 47, 37, 61]`, whose arithmetic mean is `330/8 = 41.25`.
The saved query targets are `[21, 41, 27, 3]`. Absolute errors are
`[20.25, 0.25, 14.25, 38.25]`, so MAE is `73/4 = 18.25`.
Squared errors are `[410.0625, 0.0625, 203.0625, 1463.0625]`;
RMSE is `sqrt(2076.25/4) = 22.782943181248555`, in original target units.
The linear prediction for the first query is `23.145571231301833`; the other
four fixture predictors return `41.25`.

Inspect `<task>-results/<predictor>-1/output/prediction.json`:
`evidence.fit_ids` and `evidence.preprocessing.fit_ids` must match the allowed
context IDs in `<task>-prepared/manifest.json`, never the query IDs. The simple
predictor has no learned preprocessing; its preprocessing fit list is empty.
The fit spy tests check both preprocessing and estimator fit calls through the
public module entry point. Numerical arrays use `numpy.load(..., allow_pickle=False)`.

## Identity and evidence boundaries

JSON encoding is versioned, sorted, compact UTF-8 without NaN. Row identity hashes
`[row-v1, source-file-sha256, authored-json-rows-v1, original-zero-based-position]`.
Identity is assigned before invalid-target, duplicate and conflict exclusions.
Sampling ranks `[rank-v1, seed, canonical-row-id]` by SHA-256, then row ID.
Classification allocates one per class then minimizes incremental squared error
against eligible-source proportions, with declared-class-order ties. Queries
reserve one context row per class and total context capacity. There is no seed search.

Preparation binds source bytes, parser, schema, exclusions, ordered IDs, seeds,
support, limits and scoring hash. Cache identity additionally binds predictor
recipe, implementation file hashes, Python binary/version and dependency versions.
Resolved estimator defaults and fitted numeric/category statistics are retained.
`read_attempt(path, expected_identity)` verifies all saved file hashes, exact
identity and float64 score recomputation. A mismatch raises and never overwrites
or retries. Failed attempts remain inspectable and cannot be reused as successes.
The integrity receipt detects corruption; it is not a cryptographic signature
against a party who can rewrite every file and the receipt. Retain the expected
identity/receipt independently when transferring bundles.

The numerical tolerances are 1e-6 absolute for probability bounds/sums,
1e-6 absolute and 1e-5 relative for query invariance, and 1e-12 absolute and
1e-10 relative for score recomputation. Scoring never repairs probabilities.
Query reversal and singleton chunks are checked for every successful fixture
attempt. Separate synthetic tests cover 256 queries and three 256-row contexts.
Those tests establish dimensions and sampling, not runtime feasibility at scale.

The aggregation API preserves draw scores, mean/min/max, dataset boundaries and
task/partition groups. It computes descriptive gaps and the zero-denominator
INCOMPLETE case; it never averages raw RMSE across target units. No automatic
retry, tuning, selection or training integration exists.

## Remaining evidence and implementation limits

- The sandbox's cgroup filesystem is read-only. The retained timeout and memory
  probe receipts are NOT_RUN after fail-closed preflight. Capability inspection
  and preflight rejection do not establish enforcement. Run the retained probes
  in a delegated hierarchy before accepting enforcement. The harmless memory
  probe uses a 64-MiB cap, not an 8-GiB allocation; production attempts always use
  the specified 8-GiB/no-swap/one-CPU/one-thread/five-minute caps.
- scikit-learn 1.9.1 rejects an entirely missing numeric context column in
  histogram boosting (`window shape cannot be larger than input array shape`).
  This is retained as a failed attempt, without imputing the tree feature or
  changing the approved recipe. More than 255 context categories is UNSUPPORTED.
- The TabPFN package/wheel/weight pins and approved estimator settings are recorded
  in its isolated contract. Its pinned release's internal preprocessing recipes
  remain PENDING_SOURCE_VERIFICATION. An actual TabPFN loading adapter, separately
  provisioned environment compatibility, and real query invariance are not
  implemented/verified by the mean/frequency fixture substitute. These are
  outstanding implementation/verification work, not evidence supplied by this PR.
- The learner backend interface is exercised with a labeled fixture and accepts
  explicit implementation/environment/settings/checkpoint identity. There is no
  trained r0 loader in this checkout; real comparisons remain blocked.
- Real-source preparation is deliberately refused. Admission, revised #25
  acceptance, freeze, learner selection, access/budget receipts and guarded
  execution remain external gates. No boolean flag turns a fixture into a real run.

This change is not full acceptance of #26 while the TabPFN adapter/source recipe
and resource enforcement evidence above remain outstanding.

## Executed checks

[fixture-receipt.json](fixture-receipt.json) retains the fixture summaries,
resource-preflight failures, module implementation hashes and test counts.
Baseline: `pytest --ignore=tests/comparison`, **143 passed**. Final: **184 passed**,
including **41 comparison tests**. No tests were skipped or deselected in the
final suite. All four import contracts passed. Strict type checking reported
zero errors with the explicit-path workaround below. No time-column behavior changed.

Ordinary `UV_CACHE_DIR=/tmp/r0-uv-cache make check` fails at pyright in this sandbox:
Node's synchronous interpreter probe returns `EPERM`, so pyright cannot discover
installed NumPy/pytest. The same failure existed before implementation. Python
imports those packages successfully. This workaround supplies the existing
site-packages path without changing strictness, installing dependencies, altering
repository check scripts, or suppressing diagnostics:

```sh
python3 - <<'PY'
import json, os, sysconfig, tomllib
from pathlib import Path
root = Path.cwd()
config = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['pyright']
config['include'] = [os.path.relpath(root / p, '/tmp') for p in config['include']]
config['extraPaths'] = [str(root / 'src'), str(root / '.venv/lib/python3.12/site-packages')]
config['pythonVersion'] = '3.12'
Path('/tmp/r0-comparison-pyright.json').write_text(json.dumps(config))
Path('/tmp/comparison-verify.py').write_text('''import subprocess, sys
args = sys.argv[1:]
if args[0] == 'pyright':
    args += ['--project', '/tmp/r0-comparison-pyright.json']
raise SystemExit(subprocess.call([sys.executable, 'scripts/run_check.py', *args]))
''')
PY
UV_CACHE_DIR=/tmp/r0-uv-cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 make check VERIFY='python3 /tmp/comparison-verify.py'
```

That full `make check` invocation passed; unmodified default-environment
verification remains blocked as described. Source and test paths and all strict
rules are preserved. For another Python environment, explicitly substitute its
installed-package path and interpreter version.

`make smoke` and then `make eval-smoke` were attempted after the checks. This
checkout has neither the `train` nor `eval` CLI command (T7/T12 are placeholders).
Both commands exited 2 at argument parsing. Smoke loss: **NOT_RUN**.

| eval-smoke task | metric | value | status |
| --- | --- | --- | --- |
| dev suite | unavailable | NOT_MEASURED | eval command not implemented |

The comparison walkthrough is a separate offline fixture smoke; it is not a
replacement training loss or protected evaluation-suite result.

## PR 68 review corrections

The review added shared predictor-input validation before backend fitting, shared
probability validation, rejection of nonfinite computed regression scores, and
one fitted estimator for all query-order/chunk checks. Draw summaries now reuse
the common finite-score aggregation helper.

Regression evidence: eight new boundary cases failed before correction (one
existing rejection case passed); the estimator-fit test separately observed six
fits where one was expected. All ten review cases pass after correction.

Reproduce the final checks with the explicit Pyright-path setup above:

```sh
UV_CACHE_DIR=/tmp/r0-review-uv-cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 make check VERIFY='python3 /tmp/comparison-verify.py'
```

This full invocation passed **194 tests**, all four import contracts, strict type
checking, formatting, lint and repository checks. The normal Pyright discovery
limitation remains; no check strictness was reduced. The workflow must rerun
index size checks after staging; this review leaves changes unstaged.

The walkthrough was rerun on the corrected implementation:

```sh
UV_CACHE_DIR=/tmp/r0-review-uv-cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python3 scripts/run_check.py python scripts/comparison_walkthrough.py \
  --out /tmp/r0-pr68-review-walkthrough
```

Binary, multiclass and regression fixture checks passed. Each comparison remains
INCOMPLETE. Timeout and memory probes remain NOT_RUN because the cgroup filesystem
is read-only. TabPFN 2.2.1 source/provisioning and its loading adapter are still
outstanding; cached newer releases cannot establish its pinned behavior.

After `make check`, `UV_CACHE_DIR=/tmp/r0-review-uv-cache make smoke` and then
`UV_CACHE_DIR=/tmp/r0-review-uv-cache make eval-smoke` each exited 2 because the
respective CLI command is not implemented. Smoke loss: **NOT_RUN**. The eval-smoke
table above remains applicable. No cloud execution, downloads, real inference,
protected evaluation changes or UI changes were made.
