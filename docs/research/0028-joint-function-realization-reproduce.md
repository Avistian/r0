Portable additive appendix for the component review at SOURCE `c1ee89e766194993c1a79fcc7400f7b2b48fefbe`, BASE `4c7c937cc67659d6bbb3365f12da749fb60705d4`. The original bundle remains unchanged, including its manifest.

Run `bash /path/to/appendix/RUN.sh` with `R0_REVIEW_ROOT` set to the reviewed checkout. The launcher derives its bundle from its own location, so the current directory can be outside both the checkout and bundle. If omitted, R0_REVIEW_ROOT defaults to the launcher's working directory. It is resolved to an absolute physical path. Existing UV_PROJECT_ENVIRONMENT and UV_CACHE_DIR overrides are honored; the environment defaults to the checkout's existing `.venv`, and the cache defaults to the user's uv cache. No dependency installation occurs.

R0_REVIEW_INPUTS defaults to this appendix's relocated `inputs/`. R0_REVIEW_OUTPUT must name a nonexistent directory; omission creates a fresh temporary directory. Relative inputs/output paths resolve from the launch directory. Existing output paths reject before any writes. The wrapper records these paths and environment choices in `wrapper-environment.txt`.

Exact publication validation command, executed from `/tmp`:

```sh
R0_REVIEW_ROOT=/tmp/r0-28 \
R0_REVIEW_INPUTS=/tmp/r0-28-function-realization-spec-c1ee89e-portable/inputs \
R0_REVIEW_OUTPUT=/tmp/r0-28-function-realization-spec-c1ee89e-portable/publication-output \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
bash /tmp/r0-28-function-realization-spec-c1ee89e-portable/RUN.sh
```

The wrapper checks formatting and Ruff using the supplied checkout's pyproject.toml before executing `probe.py`, `mlp_probe.py`, `adversarial_probe.py`, and `regression_probe.py`. `publication-output/` retains final raw outputs and compact JSON fixtures. The wrapper and all four programs exited 0. Programs are the exact final executed bytes intended for publication; each code/text artifact is at most 20,000 bytes.

Identity cannot be relaxed with a path override: exact HEAD, every tracked Python source digest, package import location, primary-source digest, and fixed pin-inventory hashes remain required. Source pins are split across two bounded JSON files; their merged entries reconstruct the original source-pins.json bytes exactly. PYTHONOPTIMIZE is unset, and source identity failures explicitly raise RuntimeError. Appending a newline to a copied pin file produced the expected exit 1, even with PYTHONOPTIMIZE=1 supplied. Reusing publication-output also produced the expected exit 1. These are negative checks, not product failures.

`history/preformat-programs/` reconstructs the initial portable edits deterministically from untouched original program bytes and the recorded transformations; it was retained after the first formatting command. Formatting those files with the project configuration exactly reproduces every final program in `programs/`; `history/formatted-verification/` and raw formatting logs preserve that check. `final-output/` preserves the earlier successful portable execution before the final explicit-config wrapper revision.

Changes are path handling, stronger identity rejection, splitting the source-pin inventory to meet the file cap, compact JSON output, and Ruff formatting. Numerical oracles and acceptance assertions are unchanged. ORIGINAL-SHA256SUMS and ADDITIVE-MANIFEST.json link the appendix to the original bundle; SHA256SUMS covers this appendix. No product source edit, branch change, upstream fetch, environment installation, cloud call, or new scientific claim is included.
