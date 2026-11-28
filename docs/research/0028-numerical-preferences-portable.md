# Portable numerical verifier rerun

PASS_BOUNDED: the same 708 checks passed, with identical per-category counts to the original run. The original package's SHA256SUMS still verifies. No source edits or full tests were performed by this reviewer.

Program: `programs/verify_numerical_preferences.py` (14544 bytes).
SHA256: `0c0baeec0df37b6976834ab64ddb2304ee4b8f1b8b84b31685c56d45cd59bf3b`.

Command:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r0-28/src /home/avist/Projects/r0/.venv/bin/python /tmp/numerical-preferences-spec-593c924/programs/verify_numerical_preferences.py --repo /tmp/r0-28 /tmp/numerical-preferences-spec-593c924/run-2
```

The program has no hardcoded temporary paths, accepts `--repo` (default: current directory), requires a new output directory, verifies HEAD and the complete tracked source tree, and records actual import paths. It requires the pinned Git checkout and does not claim standalone archive admission.

Verified HEAD: `593c924f5e2ea8da31d4c34d9da430aef4b38489`. Source tree: `6ee717367e397470c3112066aed42982b2993fe4`; 134 source files checked before and after execution. Imported r0: `/tmp/r0-28/src/r0/__init__.py`. Imported public API: `/tmp/r0-28/src/r0/modules/priors/__init__.py`.

`run-2/probes.jsonl` retains all 708 outcomes, including request/unit/expected/validated-value/float-hex evidence for 633 deterministic arithmetic probes, actual sampler inputs/expected/outputs for five quantile probes, and analytic expected/observed Beta moments. `run-2/beta-draw.json` retains all 4096 Beta units/values. Arithmetic probes use independent oracle values and observe public NumericalDraw validation; they are not presented as fresh sampler outputs. `run-2/source-manifest.json`, `receipt.json`, and `SHA256SUMS` retain identity and hashes.

Ruff check and format pass using `/tmp/r0-28/pyproject.toml`. Publication of large raw JSON files may split them without changing the verifier or claiming extra checks. Categorical/profile integration and whole-issue acceptance remain outstanding.
