# Joint matrix and Gaussian-mixture point plans

This checkpoint connects all five source matrix families to the
[shared sampling context](issue-28-joint-sampling.md) and the frozen
[activation/weight plans](issue-28-joint-plans.md). It implements the matrix and
kernel-mixture portion of the [design](research/0028-joint-matrix-plans.md).
The [accepted #28 specification](specs/0028-causal-network-generators.md) remains
the full scope; this is not a jointly sampled graph or whole-issue acceptance.

## Public operations

`JointMatrixPrior` declares ordered top/simple catalogs, vector family and batch
policy. Defaults contain Gaussian, importance, factorized, kernel and transformed
families; the simple catalog omits transformed. A singleton Gaussian top catalog
does not narrow the simple catalog. Custom subsets are explicit configurations;
consumers sharing a selector name must use the same catalog under existing
context rules. Both polynomial/uniform vectors and both weight batch policies
remain available.

`sample_joint_matrix(context, name, rows=..., columns=..., batch=1, prior=...)`
returns `JointMatrixPlan`. It freezes all named decisions transactionally. Family
selection uses `random_matrix_type` or `simple_random_matrix_type`, with existing
version labels retaining the source class order. Importance owns `batch * rows`
weight vectors of column width; factorized owns `batch` vectors of width
`min(rows, columns)`. Transformed matrices sample the entire simple child before
the plain activation. They never add standardization/rescaling to that activation.

`sample_joint_points(context, name, count=..., dimension=..., vector_family=...,
scalarization=...)` returns the kernel's `JointPointPlan`. This version represents
Gaussian-mixture points only. It requests `random_gmp_n_centers` with log/floor
conversion into 1..15, then mixture weights, then factor-column weights. Mixture
and factor batches share the outer weight preference names. No `kernel_point_law`
or covariance selector is introduced. A kernel matrix owns one point cloud with
`batch * (rows + columns)` points in dimension three, followed by
`random_kernel_matrix_dist_factor` in log [0.1,10]. Source arrays with incompatible
global shapes/bounds reject under the explicit strict policy; scalarized quantile
remapping remains a declared adaptation.

Every plan checks exact fields, geometry, ownership paths, branch event order,
source requests and recorded values. Irrelevant branch fields and transformed
grandchildren reject. Checked `joint_point_plan_from_dict` and
`joint_matrix_plan_from_dict` preserve these bindings. The tree fixes distinct
child paths, including event-free uniform weights; a future graph registry must
still prove unique coverage across all independently supplied plans.

## Realizations and source ordering

`materialize_joint_points(plan, decisions, rng=..., limits=...)` returns the
existing `KernelPointCloud`. Raw order is mixture weight batch, assignments,
centers, Gaussian factors, factor-weight batch, normal scale mean, signed normal
scale coefficient, component scale draws, and point innovations. Assignments use
the realized mixture masses, not outer categorical preferences. Zero total mass
rejects without retry. Factor weights multiply columns; logarithmic scaling is
`exp(mean + coefficient * component_draw)`. Existing stable arithmetic is reused.

`materialize_joint_matrix(plan, decisions, rng=..., limits=...)` returns
`JointMatrixRealization`. The record holds a complete tuple of existing
`NormalizedWeights` constructions and, for kernel branches, exactly one shared
point cloud. Its `materialize()` returns the matrix batch.

| Branch | Raw realization order and arithmetic |
| --- | --- |
| Gaussian | Full Gaussian batch. |
| Importance | Full weight batch, then Gaussian batch; multiply and normalize rows. |
| Factorized | Full Gaussian U batch, diagonal-weight batch, full Gaussian V batch; `(U * diagonal) @ V`. |
| Kernel | One shared cloud; each member takes its own contiguous left/right slice; signed Laplace kernel with Euclidean distance. |
| Transformed | Complete simple-child realization, plain activation across all flattened entries of each member, then full `1e-3` jitter batch. |

Every branch then adds a fresh full `1e-6` jitter batch and normalizes each row by
`1e-6 + L2 norm`. Importance therefore retains two normalizations. Transformed
retains child, transformation and outer jitter independently. Kernel member records
omit duplicate `point_construction`; the enclosing realization validates each
member's geometry against the one shared cloud. Transformed kernel children keep
the same shared-cloud ownership. Matrix values and their construction constants,
primitive and weight hyperparameters are bound to the plan on construction.

Named decisions are sampled before raw realization through separately injected
streams. This two-phase NumPy/float64 adaptation preserves the stated branch laws
and raw ordering, not source Torch/NumPy RNG identity or source interleaving.
Both realization helpers stage the caller RNG and commit only after structural
validation and finite arithmetic succeed. They do not silently retry failures.
The caller must coordinate named and realization streams for a complete graph;
these separate operations are not yet a graph-wide transaction or stream manifest.

## Admission and decoding

Matrix admission counts the whole realized batch and any shared cloud once, plus
the existing conservative geometry/buffer allowance. Point admission covers all
mixture components, factors, weights, assignments and innovations. Local child
operations retain their own gates. These estimates are not measured RSS, exact
interpreter allocation bounds, or graph-wide admission of the ledger, all plans,
ancestors and simultaneous calibration buffers. `parameter_count` on a plan reports
its numeric metadata; on a realization it reports retained constructions/cloud.

`joint_matrix_realization_from_dict(data, decisions=..., limits=...)` first bounds
JSON traversal and depth, validates the plan and admits its declared batch. It
then checks every matrix/vector/cloud array shape before converting arrays into
retained tuples. Unknown fields, incompatible versions, object/string arrays,
incorrect branch payloads, duplicate member clouds, activation parameter maps and
recursive transformed payloads reject. Final construction validates weight
parameters, distances, primitive, stage constants and actual cloud slices, then
checks finite arithmetic. Caller-owned limits cannot be replaced by payload fields.

Legacy recipes, samplers and decoding contracts remain unchanged. Existing
`MatrixCausalNode` accepts each realized member, allowing actual generation and
trace inspection through the authored recipe API. Such an authored recipe contains
its member construction; full shared-ledger/cloud provenance still belongs to the
separate joint realization until the new graph recipe is integrated.

## Development evidence and remaining work

Eighteen focused public cases pass in the existing Python 3.12.3 environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_joint_point_plans.py \
  tests/priors/test_joint_matrix_plans.py
```

Cases include all five branches, transformed-child ordering, shared preference
identity, literal Gaussian/importance/factorized/Laplace/flattened-softmax
arithmetic, point factor orientation/raw ordering, JSON replay, actual generated
scores and trace checks, and rollback after partial kernel planning. A two-member
2x3 Gaussian batch retains 28 construction parameters; the local gate requires
34 parameters and 6,912 estimated bytes. Exact limits pass; either one-below
parameter or byte budget rejects without changing RNG state.

Intermediate RED/GREEN, Ruff and typecheck logs are retained locally under
`docs/evidence/issue-28/joint-matrices/`. The combined focused command also includes
the four previous joint-plan/context test files: 44 cases pass. Type checking
reports zero errors. Frozen source is
`10d9b6c55107ff55c7e710f9adc01f96fc3c491b`, compared with base
`1b214e44215e48ce964ea28818b1c5a14c9c14ca`.
[Standards review](research/0028-joint-matrices-standards-10d9b6c.txt) found no
documented-rule violations and one nonblocking repeated-family-switch heuristic.
Full `make check` passed **853 tests, one skipped**, including all static gates;
pytest took 452.53 seconds. The full command uses the environment above with
`make check` replacing the focused invocation.
[Raw output](research/0028-joint-matrices-check-10d9b6c.txt) and
[author archive metadata](research/0028-joint-matrices-development.json) retain
25 development/check/report/help files. The archive contains per-file lengths and
hashes in `FILES.json`; every entry was read back and verified after packaging.
Raw logs preserve intermediate RED/GREEN and tooling failures unchanged.
[Publication gates](research/0028-joint-matrices-publication-gates.txt) pass
checkout/hooks, tracked-file/artifact limits, formatting, lint and file-token caps.
The documentation and verifier publication preserves frozen source/tests.
The [CLI help](research/0028-joint-matrices-cli-help.txt) still lacks train/eval:
conditional `make smoke` and `make eval-smoke` are NOT_AVAILABLE.
Independent specification review: **PASS_COMPONENT**, with 1,972 assertions and
341 expected rejections. Its 192-configuration grid covers all families, all four
simple transformed children, vector/batch policies and sampling modes: 188 cases
succeed and four incompatible strict global kernel-array cases reject as expected.
Literal arithmetic checks source raw order, factor-column orientation, signed log
scales, distances, flattened activation and all jitter stages. Eight retained
branch fixtures reach generation, recipe JSON and numerical traces for every
member. Exact/one-below gates cover every branch tree, plus late rollback.
[Report](research/0028-joint-matrices-spec-report.txt),
[totals](research/0028-joint-matrices-spec-totals.json),
[commands and environment](research/0028-joint-matrices-spec-reproduction.txt).

All 162 fresh legacy comparisons match the base byte-for-byte: 159 completed
results and three unchanged numerical failures. Two verifier runtime mistakes
(a preference-field name and refresh semantics outside meta mode), initial Ruff
failures and their program histories remain retained, separately from product
evidence. The seven programs in `scripts/verify_joint_matrices/` match the exact
final formatted and executed bytes. Imported modules are fingerprinted per run;
the verifier uses isolated source exports rather than the working checkout.

## Retained reproduction

[Archive metadata](research/0028-joint-matrices-spec-artifacts.json) pins both
archive parts and the original 2,560-entry manifest. There are 667 retained
non-source files, with 173 duplicate paths restored by a map. Source/base tarballs
and exports are reconstructed from pinned Git objects and checked against their
original byte hashes. From this repository, restore to a new directory:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-joint-matrices-spec-artifacts.json \
  /tmp/joint-matrices-recovered --repo "$PWD"
```

Recovery was executed and every original manifest entry verified;
[receipt](research/0028-joint-matrices-spec-recovery.json). This verifies retained
artifacts, not fresh behavioral execution. Follow the restored `REPRODUCE.txt`
and `RUN.sh` for a fresh run in a new directory, preserving the retained outputs.
The existing Python/NumPy/Torch environment is deliberately reused. Cached source
excerpts retain qualified provenance without a fresh upstream-byte claim.

## Remaining scope

Required remaining work includes covariance/uniform/ball/Gaussian root-base plans
and the RandomFunction hook; the ordered raw-realization/stream manifest; complete
graph registry and aggregate admission; ancestral fitting against finalized parent
states; graph recipe/profiles and workbench/local/Colab delivery. The wider accepted
source families/options, typed states, interventions, GPU, scale and scientific
evidence remain required. No browser, GPU, hosted or predictive claim is made here.
The [root/function design](research/0028-joint-root-function-plans.md) preserves all
eight function families, five cheap product-child choices, duplicate selector
slots, source orientations and degenerate count cases. Its proposed checks are
NOT_RUN; the note does not mark any of those integrations complete.
