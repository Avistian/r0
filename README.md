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

## Data attributions
The planned benchmark suite uses RelBench databases, each under its own terms:
rel-f1 (CC BY-NC-SA 3.0), rel-stack (CC BY-SA 4.0), rel-trial (ClinicalTrials.gov terms).
No data is stored in this repo.

## Licence
Apache-2.0 (see LICENSE).
