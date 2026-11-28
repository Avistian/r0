# Reproduce the repair checks

Select a source checkout or frozen archive with PYTHONPATH and use a fresh output directory for every command. These programs use public interfaces only; no dependency installation is required.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/source/src /home/avist/Projects/r0/.venv/bin/python shared_preference_preserve_extreme.py /new/extreme
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/source/src /home/avist/Projects/r0/.venv/bin/python shared_preference_boundaries.py /new/boundaries
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/source/src /home/avist/Projects/r0/.venv/bin/python shared_preference_numeric.py /new/numeric
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/source/src /home/avist/Projects/r0/.venv/bin/python shared_preference_matrices.py /new/matrices
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/source/src /home/avist/Projects/r0/.venv/bin/python shared_preference_legacy.py /new/legacy
```

The receipt program accepts `--repo`, `--commit`, `--runs`, `--baseline` and `--output`. `--runs` contains the five named output directories above, `source.tar`, and the portable scripts; `--baseline` contains an earlier legacy snapshot with its per-case records. It checks all imported r0 modules come from the selected archived source, verifies every tracked source file against the pinned Git blobs, independently compares all 504 legacy outcomes and records script hashes. It also checks the retained log-probability oracle errors with an absolute tolerance of 2e-13.

Original `47630e8` rejection and verifier harness errors are retained in the parent artifact directory. The repaired programs and outputs never overwrite them. No whole-issue, browser, Colab, GPU, full-suite or author-census acceptance follows from these bounded checks. Historical backend paths embedded in saved files identify the actual run and are not portable replay destinations; the scripts create fresh destinations from their output argument.
