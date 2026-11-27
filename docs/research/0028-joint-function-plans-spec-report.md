# Independent spec review: PASS_COMPONENT

BASE `731f6a44b309a75fc5c5f756168550eac48b372a`; reviewed SOURCE
`a6ab8c70902d9d474cea12e7cab2aadf67ca35f5`. Source checkout remained read-only;
preflight and final status show only the pre-existing untracked `.venv`.

No missing/partial requirement, scope creep, or incorrect implementation found
within this named-sampling/typed-plan/strict-JSON/ledger-binding component.
The reviewed design requires: “Preserve duplicates with distinct slot IDs”; “Entire
first cheap selector/subtree, then second cheap selector/subtree”; “Derive exact
expected events/catalogs/values/shapes on decode”; and “Before conversion bound
bytes/depth/nodes/containers/strings/paths/numeric leaves/axes.”

Independent public-API probes passed 706 assertions across 96 plans (all eight
families, both GP spectra, all five cheap children), source-derived request and
numerical laws, sharing modes, MLP stage ordering, discretization edge cases,
changed values/requests/ledger, malformed/deep arrays, exact numeric/byte metadata
boundaries, and named/nested failure rollback. Supplement passed 444 exact scalar
comparisons against the base revision, 16 ownership/geometry mutations, and all
five matrix-family round trips. Both final executed scripts are Ruff-clean and
under 20,000 bytes.

No current upstream refresh was performed. Attribution uses only cached I2func.py
and _base.py; hashes/import paths/source identity are retained in identity.json.
Realization, numerical frozen evaluators, fitting, raw provenance, roots, graph
transactions/integration, whole issue #28, browser/Colab/GPU/scientific/scale
acceptance are not established by this component verdict. No full test suite,
dependencies, downloads, cloud, or secrets were used.

## Reproduce

From `/tmp/r0-28`, run each command separately (60-second cap):

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv timeout 60s python3 scripts/run_check.py python /tmp/r0-28-function-plans-spec-a6ab8c7/probe.py
UV_CACHE_DIR=/tmp/r0-28-uv-cache UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv timeout 60s python3 scripts/run_check.py python /tmp/r0-28-function-plans-spec-a6ab8c7/ownership_legacy.py
```

## Retained harness history

`probe-initial.py/.log`: oracle incorrectly expected weight roles `decay/scale`
instead of the existing scalarized `decay:0/scale:0`.
`probe-boundary.py/.log`: ordinary binary log interpolation at exact midpoint
produced a floor differing from the established high-precision baseline. The
final source-law comparison uses away-boundary units; the separate base-revision
compatibility probe includes exact boundaries and confirms unchanged behavior.
These were probe defects, not product defects. Original scripts and raw failures
are retained rather than overwritten.

Final raw outputs: `probe.log`, `ownership_legacy.log`; fixtures: `fixtures.json`,
`ownership-fixtures.json`; hashes: `identity.json`, `supplement-identity.json`;
review diff: `review.diff`; Ruff: `ruff.log`, `ruff-supplement.log`.
