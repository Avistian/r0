# r0

A small tabular foundation model (TabPFN-style), pretrained on synthetic data, built by a
personal software factory. Research code: no guarantees, not for production.

## Usage

Requires Python 3.12 and uv. From the repository root:

```sh
make setup && make check
uv run r0 --version
```

`uv run` runs `r0 --version` in the project's virtual environment.

## Verification environments

`make setup` explicitly syncs this checkout's `.venv` and activates its commit hook.
Verification commands (`make check`, its component targets and the commit hook) use
`scripts/run_check.py`. They require an existing environment, run uv in offline, no-sync mode, and select this checkout's `src` directory. Verification never
creates an environment or repairs missing dependencies; run setup explicitly when needed.

To deliberately reuse an existing environment without changing its editable installation:

```sh
make hooks
UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv make check
```

The wrapper checks path files, editable finders and direct-URL metadata before and after
each command. A change fails verification and names the affected files; it is not silently
rolled back. This detects editable-install changes, not arbitrary writes by tested code.
An inherited active environment alone does not select reuse. The default remains `.venv`.
For other read-only verification commands, use `python3 scripts/run_check.py COMMAND ...`;
a bare `uv run` can sync an environment and is outside this guard.

`make preflight` checks source resolution and local hook activation. `make tokens` and the
commit hook run the same token checker over **all Git index blobs**, including JSON and
filenames containing spaces. Staged oversized content remains a failure even if its working
copy is replaced or deleted. The existing binary, lockfile and workflow exclusions remain.
Stage edits before checking the intended commit. Both paths also enforce the 1 MiB index
artifact limit. CI runs the same `make check` after explicit frozen environment setup.

## Predictive reference walkthrough

Issue #22 provides bounded CPU beta–Bernoulli, Dirichlet–categorical and linear/RBF GP
references. See [the worked reference cards](tests/fixtures/oracles/README.md) for
independent calculations, metric conventions and replay instructions. The
[accepted specification](docs/specs/0022-oracles.md) defines the scope.

```sh
python3 scripts/run_check.py python scripts/oracle_walkthrough.py --out /tmp/r0-oracle-evidence
```

## Generate and save datasets without the explorer

Run a bounded CPU generation job using an existing recipe:

```sh
python3 scripts/run_check.py r0 generate --config configs/generation/polynomial-cpu.toml --out /tmp/r0-datasets
```

The command prints a JSON completion summary. It saves dataset ZIP groups and
receipts under the output directory; repeat the same command to resume without
regenerating completed tasks. A timeout or incomplete run exits nonzero and keeps
completed groups. `--quiet` suppresses the summary but preserves errors. The
configuration sets task count, seed, shape, wall-time and storage limits. Other
examples live in `configs/generation/`. A `.json` config can instead contain the
public serialized `RunRequest`, including retained structural generator recipes.
Decoding and replay execute inside the timeout-protected worker.

Reload the saved tasks through the public API:

```python
from pathlib import Path
from r0.modules.generation import load_group

for archive in sorted(Path("/tmp/r0-datasets").glob("groups/**/*.zip")):
    for task in load_group(archive):
        print(task.inputs.shape, task.targets.shape)
```

## Training generator kernels

See [bounded batch consumption](docs/bounded-batch-consumption.md) for a runnable
example that feeds generated or saved episodes into an optimizer through the
shared episode interface.

See [direct GP kernel recipes](docs/gp-kernels.md) for RBF, Matérn, rational-quadratic
and spectral-mixture configuration, numerical limits and saved-task walkthroughs.

## Data attributions
The planned benchmark suite uses RelBench databases, each under its own terms:
rel-f1 (CC BY-NC-SA 3.0), rel-stack (CC BY-SA 4.0), rel-trial (ClinicalTrials.gov terms).
No data is stored in this repo.

## Licence
Apache-2.0 (see LICENSE).

## Local execution evidence

Keep generated logs, receipts, screenshots and run outputs under `docs/evidence/`,
which is ignored by Git. PRs should include concise results and reproduction
commands. Authored fixtures required by tests and walkthroughs remain versioned
under `tests/fixtures/`; they are inputs, not generated run outputs.

Historical issue #28 artifacts formerly under `docs/research/` are retained locally
in ignored `docs/evidence/issue-28/legacy-research/`. See the
[evidence retention note](docs/issue-28-evidence-retention.md) for the migration.

For a spatial causal graph and matching classification view, see the
[runnable spatial examples](docs/issue-28-spatial-functions.md).
A [high-frequency sinusoidal example](docs/issue-28-atomic-delivery.md#high-frequency-sinusoidal-mechanism)
also generates saved datasets directly from a JSON request.

## Paired regression and classification targets

`generate_dual_targets` generates one table and exposes its original continuous
labels alongside threshold-based class labels. Thresholds are explicit: equality
goes to the upper class, and empty classes are allowed. No query-dependent fitting
or balancing is performed. This is the named r0 reconstruction for the dual-target
requirement; the peer algorithm's exact mapping is undisclosed.

```python
from pathlib import Path
from r0.modules.priors import LinearRecipe, generate_dual_targets
from r0.modules.generation import save_group

pair = generate_dual_targets(
    LinearRecipe(rows=128, context_rows=96),
    boundaries=(-1.0, 0.0, 1.0), seed=28, task_id=0,
)
save_group(Path("/tmp/regression.zip"), [(0, pair.regression)], max_bytes=10_000_000)
save_group(Path("/tmp/classification.zip"), [(0, pair.classification)], max_bytes=10_000_000)
```

For classification archives directly from the CLI, use
`configs/generation/dual-target-cpu.toml` with `r0 generate`. The recipe retains the
original regression generator, so both views remain reproducible from the same seed
and task ID. Any supported regression generator can supply the underlying table.
