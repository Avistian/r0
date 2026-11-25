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
