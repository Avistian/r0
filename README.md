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

When reusing an existing environment in a fresh checkout, run `make hooks` to activate
this repository's commit hook without syncing dependencies. Local `make preflight`
checks the source checkout and hook activation; CI runs the checks directly.
`make check` and the commit hook both enforce the 1 MiB limit on staged/tracked Git
blobs, including files whose working copies were subsequently changed or removed.

## Predictive reference walkthrough

Issue #22 provides bounded CPU beta–Bernoulli, Dirichlet–categorical and linear/RBF GP
references. See [the worked reference cards](docs/evidence/issue-22/README.md) for
independent calculations, metric conventions and replay instructions. The
[accepted specification](docs/specs/0022-oracles.md) defines the scope.

```sh
uv run python scripts/oracle_walkthrough.py --out /tmp/r0-oracle-evidence
```

## Data attributions
The planned benchmark suite uses RelBench databases, each under its own terms:
rel-f1 (CC BY-NC-SA 3.0), rel-stack (CC BY-SA 4.0), rel-trial (ClinicalTrials.gov terms).
No data is stored in this repo.

## Licence
Apache-2.0 (see LICENSE).
