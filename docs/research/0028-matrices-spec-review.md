# Spec review

PASS for the matrix-family checkpoint `7eea0bd5ab6f553aa07f014ed5a353b4d0db69e9` against `9871fdaf7037c7ed9b09926f5073f63f5cf6ff31`. No scoped specification defect found; this is not acceptance of all issue #28.

Independent Fraction/Decimal/analytic oracles passed 224 checks covering finite Gaussian retention, positive-vector rounding before factor products, cancellation/overflow recovery, signed Laplace kernels, jitter normalization, flattened activation, point formulas and assignments, raw-stage rejection, budgets, geometry disagreement and bounded decoding. Sampled record scalar counts are independently computed from retained numeric leaves. Array admission combines retained RNG/admission traces with source inspection.

The 72-recipe profile/family/task/seed matrix completed 4,505 assertions: 71 tasks generated and replayed; one mixed-profile forward multiclass seed-1 draw retained its explicit square-activation overflow failure. All five raw families, four transformed children, three point-law/base combinations and widths 1/2/4 appeared. Six real CPU backend flows (both profiles through all graph families) generated, traced, saved, reopened and replayed; maximum trace target error was zero. This is not browser evidence.

All 378 cases across the six previous profiles matched the baseline exactly, including two failures. Both legacy manifests have SHA256 `4e9d529d8022f8516265d163ec73282f23f861493b3f60114b95da9abe1a4d16`.

`run-1` is retained as HARNESS_ERROR: a zero coefficient correctly bypassed the exact factor fallback. The corrected probe and complete successful run are `run-2`; no candidate change occurred. `run-2/receipt.json` records immutable git archives, exact subprocess commands, imports and script hashes. `final-check.json` confirms final script bytes match executed bytes and Ruff passes using the repository config and cwd. Source HEAD and status stayed unchanged.

Primary-source differences and remaining full-issue scope are recorded in `source-review.md`. Browser, Colab, GPU, source joint-prior and scientific acceptance remain outside this checkpoint.

# Reproduce

The four sibling scripts in `programs/` are standalone and each is below 20,000 bytes. They require an existing Python environment with this repository's dependencies and a local Git repository containing the pinned revisions; no retained `/tmp` data are inputs.

Executed from `/tmp/r0-28`:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/avist/Projects/r0/.venv/bin/python /tmp/matrix-spec-7eea0bd/programs/verify_causal_matrices.py /tmp/matrix-spec-7eea0bd/run-2 --repo /tmp/r0-28
```

For another machine, replace Python/script/repository paths and supply a **new nonexistent output directory**. Candidate/base defaults are pinned in the runner; optional `--candidate` and `--base` override them. The runner creates read-only git archives, selects each archive with PYTHONPATH, disables bytecode, runs numeric and case probes, and compares the complete legacy outcomes. It does not fetch, install or modify the repository/environment.
