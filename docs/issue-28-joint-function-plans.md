# Issue 28: joint source function plans

This component extends shared named sampling with inspectable plans for all eight
source RandomFunction families. It is one step toward the full
[accepted scope](specs/0028-causal-network-generators.md); it does not close #28.

## Scope and source law

The source contract and cache attribution are in
[the root/function design](research/0028-joint-root-function-plans.md).
The implementation adds `JointFunctionPrior`, `JointFunctionPlan`,
`sample_joint_function`, and `joint_function_plan_from_dict` to `priors`.

`default` expands to MLP, tree, discretization, linear, quadratic, GP, EM, product;
`tabpfnv2` expands to MLP, tree, discretization. Explicit comma-separated tokens
preserve order and duplicate slots. Whitespace is not stripped. Product children
use the fixed cheap catalog GP, tree, discretization, linear, quadratic. Each
function retains its input/output widths and independent calibration row count.

| Branch | Retained choices and owned children |
| --- | --- |
| Linear | Output-by-input normalized matrix plan, no bias |
| Quadratic | One matrix batch, one square matrix per output; dimension min(input,20)+1 |
| Tree | Rounded tree count 1..128; floored depth 1..7 |
| Discretization | Count source-case tag, distance exponent, linear matrix plan |
| GP | Tail exponent and product/radial selector; radial-only raw-factor weight plan then scale |
| EM | Component count, spread, distance p and q, output matrix with component input width |
| MLP | Width and depth; start/end flags; matrix and standard activation plans in evaluation order |
| Product | Exactly two cheap function plans; complete left subtree before right; both share input geometry |

The new integer policy `source-clip-floor-v1` reproduces floor then clip to
[low, high-1], including source randint(2,2) returning 1. The existing `floor`
policy retains its distinct-bound requirement. Numerical interpolation retains
r0's existing high-precision path: binary floating-point source values at exact
integer boundaries need not be bit-identical. Discretization with one fit row
or an explicit positive center count skips the named count request. An automatic
count with min(rows, cap)=2 retains the degenerate draw and its value 1; cap=1
with multiple rows fails transactionally. The cap can be 1 in the two branches
that do not consult it.

## Binding and resource boundaries

Plans validate the exact selector slot, request laws, scalar values, operation
ownership and event order against a frozen `JointSamplingRecord`. Matrix and
activation children reuse their existing binding validators. Scope and catalog
contracts remain shared across local/meta/global named sampling; changing a
cached name's catalog is rejected. A failed nested function transaction restores
the complete caller decision state and RNG.

JSON decoding checks caller-owned raw numeric, text, container and depth limits
before reconstruction, then rejects malformed nested objects, unknown fields,
incorrect branch cardinalities, versions, ownership and inconsistent decisions.
The raw budget counts configured prior counts and Boolean scopes as well as
realized scalars. `plan.parameter_count` instead describes geometry and realized
scalars; it excludes prior configuration and the shared ledger and is not a graph
storage bound. The shared raw metadata gate was moved without changing its law;
frozen numerical function decoding continues to use the same gate.

These plans contain named choices only. They do not allocate the represented
matrices, prove calibration ancestry, or admit complete graph memory. The prior
frequency count may describe a large future realization; a later materializer
must admit its full raw and derived arrays before allocation. Event-free uniform
children have structural ownership within these plans; graph-wide registration
and atomicity remain required.

## Validation

Source revision: `a6ab8c70902d9d474cea12e7cab2aadf67ca35f5`.
Comparison base: `731f6a44b309a75fc5c5f756168550eac48b372a`.

Focused public-interface checks: 45 passed across function plans, source integer
laws, numerical frozen functions and joint matrices. They cover all eight
function branches, all five product-child families, MLP stage flags, both GP
spectra, discretization special cases, roundtrips, raw numeric admission and
transactional rollback. Full `make check` passed: **882 passed, one skipped in 458.12 seconds**; all
static gates passed. Training/evaluation smoke targets remain unavailable because
the CLI has no train/eval commands; this is not model or scale acceptance.

Independent specification review: **PASS_COMPONENT**. Retained public probes
passed 706 assertions over 96 function plans, 444 exact scalar compatibility
comparisons with the base revision, 16 ownership/geometry mutations, and all five
matrix-family roundtrips. These are distinct counts, not independent datasets or
scientific experiments. Cached source attribution was checked locally; upstream
was not refreshed.

Independent standards review: **zero hard violations, two heuristic findings**:
repeated geometry/stage-order logic and string-keyed parameters. They remain
nonblocking maintainability concerns. Runtime request/shape validation retains
the exact source contracts; no claim of eliminating those concerns is made.

The initial rollback test rejected an incompatible cached function catalog before
the intended count-bounds failure. The repaired test first samples an unrelated
activation, then verifies rollback at the intended failing count request in all
three sampling modes. Both outputs are retained as development history.

Reproduction uses the existing declared environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_joint_function_plans.py \
  tests/priors/test_source_integer_law.py tests/priors/test_frozen_functions.py \
  tests/priors/test_joint_matrix_plans.py
```

## Remaining implementation

Raw function realization, independently fitted wrapper ancestry, construction
manifests, root bases and root attachment, graph transactions and aggregate
admission remain required. So do the accepted paper families, typed variables,
interventions, diagnostics, workbench, local and Colab delivery. Numerical frozen
function evaluation is the earlier [separate checkpoint](issue-28-frozen-functions.md).

## Retained evidence and portable replay

The [independent report](evidence/issue-28/legacy-research/0028-joint-function-plans-spec-report.md),
[standards report](evidence/issue-28/legacy-research/0028-joint-function-plans-standards.txt),
[full check](evidence/issue-28/legacy-research/0028-joint-function-plans-check-a6ab8c7.txt),
[development manifest](evidence/issue-28/legacy-research/0028-joint-function-plans-development.json) and
[independent archive metadata](evidence/issue-28/legacy-research/0028-joint-function-plans-spec-artifacts.json)
retain the inputs, raw outcomes and historical harness failures. Portable replays
repeat existing cases; they do not add coverage counts. The original artifact set
remains byte-identical within the recovered bundle's `original/` directory.

Recover the evidence into a new directory:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-joint-function-plans-spec-artifacts.json \
  /tmp/function-plan-evidence
```

Run from a checkout of source `a6ab8c7`, using the published verification scripts
(or identical recovered programs). Each command requires a new output directory:

```sh
R0_REVIEW_INPUTS=/tmp/function-plan-evidence/inputs \
R0_REVIEW_OUTPUT=/tmp/function-plan-replay-main \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
python3 scripts/run_check.py python /tmp/function-plan-evidence/probe.py

R0_REVIEW_INPUTS=/tmp/function-plan-evidence/inputs \
R0_REVIEW_OUTPUT=/tmp/function-plan-replay-ownership \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
python3 scripts/run_check.py python /tmp/function-plan-evidence/ownership_legacy.py
```

The verifier checks imported-source location and every reviewed priors source
file against the pinned revision before probing. Cached source inputs are hashed
and attributed; this does not refresh upstream. Environment paths above describe
the verified local environment and must identify an already established compatible
environment elsewhere.
