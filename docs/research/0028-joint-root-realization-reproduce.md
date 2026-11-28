# Independent root realization/emission review

Component HEAD: `78632956b1caf236ba521397d564311e2b006465`.
Comparison: `git diff 2488161135eca07374a8e433c9b76de81f573e5d...HEAD`.
Checkout: `/tmp/r0-28`; status before and after was only `?? .venv`.
The source checkout, tests and environment were not edited or synchronized.

Reproduce from the pinned checkout:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv python3 scripts/run_check.py python /tmp/r0-28-root-realization-spec-7863295/review.py
```

The program derives its bundle from `__file__` and checkout from
`R0_REVIEW_ROOT` or cwd. Every execution creates a fresh `run-*` directory.
`pins.json` pins the ten changed files and retained source inputs. It also
checks exact Git HEAD and the actual imported module path. `executed-programs.json`
pins the executed review programs. Explicit exceptions enforce all gates;
Python assert statements are not used. Final Ruff format/check logs are retained.

Evidence: `attempt-1.log` and `run-y4xagjev/checks.jsonl` (PASS). No executable
attempt failed. Initial pre-execution lint identified formatting, unused imports,
and immediately called closure bindings; those harness issues were corrected
before the executable run. The final lint evidence covers the executed bytes.

`source_points.txt` and `source_function.txt` are copies of the supplied local
cached upstream inputs, with hashes matching the parent-provided identities.
The provenance is recorded in `docs/research/0028-joint-root-function-plans.md`:
TabICL source commit `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`.
This is local cached provenance, not fresh upstream verification. Neither cached
source program was imported/executed; the independent equations are in `oracle.py`.

Coverage:

- All 32 root base/function combinations, retained calibration, frozen-function
  emission and serialization round trips. Seven function families use independent
  NumPy equations. MLP uses separate standalone-function realization equality and
  public evaluator chunking; its activation algebra is inherited acceptance.
- Root raw calibration/emission draws and covariance Gaussian-before-weight order;
  both uniform/Gaussian covariance laws and uniform/polynomial weights. Covariance
  factor is independently column-scaled. Literal A,w,B example gives (-1,-1), with
  FF^T and FF^T/3; ball radius one-half and valid origin; sphere and zero-direction
  counterexamples reject; zero covariance factor is accepted.
- Fifteen decoder corruption checks, including missing fields, bool dimensions/
  draws/radii, stale values/outputs/identity, unknown versions/fields, ragged and
  nonfinite matrices. All three pairwise shared-BitGenerator combinations reject.
- Failure injected after a real complete function fit proves raw/fit streams were
  consumed internally before all caller streams roll back. Emission fault after
  raw draws and real numerical overflow both roll back, with exact fresh recovery.
- Public-API binary search determines minimum admitted limits for one linear case:
  child/root/emission parameters 53/138/158, working_bytes 11748/32208/67264. At
  root_min-1 the child succeeds and root rejects without caller-stream movement;
  emission_min-1 rejects. These are conservative admission thresholds, not RSS.
- Chunk invariance means fixed-function evaluation of the same rows. Ball raw
  sampling is intentionally G(R,d) then U(R); partitioning sampling calls changes
  rows and is retained as EXPECTED_DIFFERENCE, not claimed to be invariant.

Limitations: bounded independent fixtures, no statistical goodness-of-fit or
process-RSS measurement. No training, paid/cloud, GPU, browser, Colab, throughput,
or scientific acceptance. Parent owns the full `make check` result. Global graph
transactions, registry/manifest, ancestry calibration and all remaining #28
integration are still required outside this component.
