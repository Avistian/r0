# Frozen function independent specification review: PASS_COMPONENT

SOURCE cbbaff2cb4c8c6e485ad3814f79e84f3c1d68967
BASE cb7977a916ff6f11d95790b2de4dd591ab8d3f39
Scope: low-level frozen numerical equations, JSON reconstruction and resource
admission only. This is not acceptance of completed issue28, its source sampler,
independent fitting, fitted-stage provenance, root/graph integration or delivery.

## Results and exact case accounting

| Final program | Cases | Result |
| --- | ---: | --- |
| check_ties.py (unchanged original) | 98 | PASS |
| check_roundtrip.py (unchanged original) | 1 | PASS |
| check_equations.py (unchanged final ed5b67a verifier) | 142 | PASS |
| reproduce_quasi_tie.py (unchanged final ed5b67a reproducer) | 1 | PASS |
| check_contracts.py | 157 | PASS |
| check_admission.py | 291 | PASS |
| check_legacy.py | 258 | PASS |
| Total final cases | 948 | PASS_COMPONENT |

These are cases across seven standalone verifier programs, not author pytest
counts, statistical samples or distinct scientific claims. Per-case JSONL retains
fixtures, expected/observed vectors, limits and rejections where applicable.
An earlier admission run also passed291 cases before a lint-only lambda binding
cleanup; it is retained separately and excluded from the totals above.
No numerical harness failure or product failure occurred at this source revision.
Prior rejected source revisions and their harness histories remain untouched.

Actual prior-r0 regression is separate: check_r0_regression.py generated18 cases
on each of BASE and SOURCE, all completed, all18 canonical records byte-equal.
The grid is three old profiles (causal-affine-v1, causal-source-standard-v1,
causal-source-standard-matrices-v1), forward/layered families, regression/binary/
multiclass tasks; seed649, taskIDs0..17, rows16/context8/features2. Public recipe
JSON reconstruction, generate_task, explain_generation, save_group, verify_group
and load_group are exercised. Every loaded task serializes identically to its
original; every saved archive member hash matches between revisions. Per-case
JSON/ZIP and exact import/environment receipts are under r0-base/ and r0-source/.
This finite18-case regression does not establish every old profile unchanged.

## Independent verification coverage

Equations use independent scalar math.fsum loops, NumPy primitives, exact integer
radicals and Decimal precision1100. Expected outputs never call the product
evaluator. Covered output-by-input linear orientation; nonsymmetric quadratic
cross/linear/constant terms and widths1/20/21 with reordered20-input subsets;
strict oblivious-tree split bits, averaging, tree/depth1/1 and128/7; finite extreme
means and cancellation; ordered discretization and p=.5/1/2/4, degenerate one
center, exact L1/quasi-distance ties, adjacent represented distances from minimum
subnormal through1e308, sparse1D/3D coordinates, overflow subtraction; RFF phase,
frequency and coefficient orientation; EM p/q extrema, scalar normalization
without an input-width multiplier, component bounds2/16/34 and extreme sigma;
all25 permitted cheap-child pairs; MLP depths1..3 and all start/end flags,
source hidden-width endpoints1/127, dimension mismatches, stage order, row-local
softmax/rank/argsort and all six authored activation parameters.

Each of142 equation cases includes JSON roundtrip, independent expected outputs,
single-row chunk evaluation, repeated calls after mutating a returned tensor,
input/frozen-record preservation and Torch global RNG preservation. Authored
calibration constants and placeholder identities are deliberately not evidence
of sampling or fitting on independent preceding-stage outputs.

The157 contract cases reject invalid/absent/unknown versions, unknown fields,
non-object roots, array strings/mappings, non-string keys, ragged/scalar arrays,
bools/nonfinite/oversized numbers, recursive payloads, traversal/numeric/text
limits, illegal nested product families, invalid geometry, MLP counts/flags/
widths, wrong input dtype/rank/width/subclass/nonfiniteness, and numerical failures.
Exceptions ValueError/TypeError/KeyError count as rejection; no stronger exception
normalization claim is made.

The291 admission cases use independently hand-derived node/width/parameter/byte
totals for all eight families plus nested composition. Exact limits pass;
one-below node count, maximum/total widths, parameters and bytes reject. Actual
rows and center-dependent working floors are checked. Decoder admission and
product/MLP aggregate work are included. Allocation/conversion tripwires require
limit rejection before Tensor.to or parameter new_tensor creation. Resource
estimates are not measured RSS, process caps or whole-graph admission proof.

The258 cached-source transform comparisons execute only cached I2func.py _transform AST bodies
with independently authored numerical state, over six input/output geometries,
all families, all25 cheap pairs per geometry and all12 MLP flag/depth combinations
per geometry. The extracted source bodies and line ranges are retained in
legacy-transform-extractions.json. No source fitting/sampling code executes. These comparisons are not prior-r0
regression or legacy byte-identity evidence; the18 paired BASE/SOURCE cases above
provide that distinct bounded evidence.
MLP wrapper arithmetic uses the independent NumPy oracle; this corroborates
source stage order, not full upstream activation calibration. Numerical fixtures
use dedicated np.random.default_rng seeds20261127 and983270, never author tests.

## Scope, provenance and remaining work

No requirement defect or scope creep remains identified within this reviewed
component. Research spec109 and component contract27 tie semantics now pass all
retained counterexamples; component JSON contract52-53 closes the accepted-record
roundtrip failure. All three historical REJECT directories remain intact:
/tmp/r0-28-functions-spec-690297b,
/tmp/r0-28-functions-spec-c58706e,
/tmp/r0-28-functions-spec-ed5b67a.

Source requests, sampled distribution/provenance, independent fitting and
stage ancestry, root attachment, graph transactions/admission, moments and full
workbench/local/Colab remain required (research134-137; accepted spec14-18,
76-80,122-130). Author owns implementation; independent verifier owns subsequent
integration acceptance. CPU was exercised; CUDA/browser/Colab/scale/scientific
acceptance NOT_RUN. Full repository checks belong to the author's separate run.

Cached I2func.py is attributed to TabICL9530bd65a7b806f6ad4f9f4335af745577e9d3cc;
local SHA256 ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c.
It was not freshly fetched or verified upstream. This limits source provenance.

## Environment, commands and reconstruction

Python3.12.3 from /home/avist/Projects/r0/.venv/bin/python; Torch2.15.0+cpu;
float64 CPU, one Torch thread. Exact NumPy/platform/executable/import paths and
hashes are in each *-receipt.json (roundtrip.json for its standalone program).
All imported r0 modules are asserted under the frozen source export. No install,
sync, downloads, cloud, secrets, merges, pushes or worktree/source/test edits.

Run `bash /tmp/r0-28-functions-spec-cbbaff2/replay.sh` to reproduce in a fresh
owned temporary directory while preserving this evidence. The driver was executed successfully; fresh replay artifacts are retained at
/tmp/r0-28-functions-spec-cbbaff2-replay-5JYVs7. All seven component programs and
both prior-r0 revision executions passed again; these repeat executions are not
added to final coverage counts. The driver copies
retained final verifier bytes and frozen source, runs Python -B, stops on failure,
and prints the replay path. Original command form for each program:
```
PYTHONPATH=/tmp/r0-28-functions-spec-cbbaff2/source/src /home/avist/Projects/r0/.venv/bin/python -B /tmp/r0-28-functions-spec-cbbaff2/PROGRAM.py
```
For actual prior-r0 comparison, run check_r0_regression.py base under
PYTHONPATH=.../base/src, then check_r0_regression.py source under
PYTHONPATH=.../source/src, in a fresh artifact root. replay.sh includes both.

Final Ruff format/check uses source/pyproject.toml and known-first-party r0,
explicitly listing verifier programs (never formatting cached source). Every
verifier program is below20KB. Manifest binds final executed program bytes,
imports, fixture/log receipts and exported source/base files.

Original archive commands from /tmp/r0-28, into an empty destination:
```
mkdir /tmp/r0-28-functions-spec-cbbaff2
mkdir /tmp/r0-28-functions-spec-cbbaff2/source /tmp/r0-28-functions-spec-cbbaff2/base
git archive cbbaff2cb4c8c6e485ad3814f79e84f3c1d68967 src pyproject.toml AGENTS.md CODING_STANDARDS.md docs/requirements.md docs/specs/0028-causal-network-generators.md docs/research/0028-joint-root-function-plans.md docs/issue-28-frozen-functions.md | tar -x -C /tmp/r0-28-functions-spec-cbbaff2/source
git archive cb7977a916ff6f11d95790b2de4dd591ab8d3f39 src pyproject.toml AGENTS.md CODING_STANDARDS.md docs/requirements.md docs/specs/0028-causal-network-generators.md docs/research/0028-joint-root-function-plans.md | tar -x -C /tmp/r0-28-functions-spec-cbbaff2/base
cp /tmp/r0-28-functions-spec-ed5b67a/check_ties.py /tmp/r0-28-functions-spec-ed5b67a/check_roundtrip.py /tmp/r0-28-functions-spec-ed5b67a/evidence_common.py /tmp/r0-28-functions-spec-ed5b67a/check_equations.py /tmp/r0-28-functions-spec-ed5b67a/reproduce_quasi_tie.py /tmp/r0-28-functions-spec-ed5b67a/I2func.py /tmp/r0-28-functions-spec-cbbaff2/
git diff cb7977a916ff6f11d95790b2de4dd591ab8d3f39...cbbaff2cb4c8c6e485ad3814f79e84f3c1d68967
```
Read-only preflight confirmed pinned HEAD and only untracked author-owned evidence
plus .venv; no live test process then. Author later launched full makecheck
separately; these independently exported checks do not include that suite.

Final read-only worktree check still showed HEAD cbbaff2 with no source/test
changes; author documentation and evidence packaging changes were present and
are outside the pinned-source specification review.
