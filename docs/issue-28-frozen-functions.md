# Frozen source function equations

This component implements the numerical records, evaluation and bounded JSON
reconstruction for the eight RandomFunction families in the
[root/function design](research/0028-joint-root-function-plans.md). It is one part
of the [accepted #28 scope](specs/0028-causal-network-generators.md).
These records accept already realized coefficients, centers, thresholds and
activation constants. They do **not** yet sample the source function prior, prove
independent fitting, or attach these functions to generated causal graphs.

## Public records and evaluation

All operations are exported through `r0.modules.priors`. The eight frozen records
contain immutable tuples of finite numbers and validate their geometry. Product
children and MLP stages are typed and bounded; arbitrary recursive functions are
not accepted. `evaluate_frozen_function(record, inputs, limits=...)` accepts a
nonempty floating Torch matrix with the declared input width, admits its geometry,
then evaluates in float64 on the input's device. It returns a new tensor and never
fits, draws randomness, updates the record or modifies the inputs. Tensor subclasses
are not supported. Input and output nonfiniteness reject explicitly without retry.

| Record | Retained equation and constraints |
| --- | --- |
| `FrozenLinearFunction` | Output-by-input matrix M; X M transpose, without bias. |
| `FrozenQuadraticFunction` | Ordered input indices, constant 1 appended last, one square Q per output; z transpose Q z. Uses all inputs through width 20, otherwise exactly 20 distinct selected inputs. Q need not be symmetric. |
| `FrozenTreeFunction` | 1..128 oblivious trees, depth 1..7, one feature/threshold per tree and depth, complete leaf tables. Strict greater-than split, depth-zero bit least significant; average trees. |
| `FrozenDiscretizationFunction` | Ordered centers and linear output map; first nearest center under p-distance, p in [0.5,4]. Targets are the frozen linear map applied to centers. No resampling/refitting of targets. |
| `FrozenRFFFunction` | Input-by-frequency matrix, phases in [0,2pi), frequency-by-output coefficients; cos(X F + phase) C. Finite Fourier approximation, not an exact Gaussian process. |
| `FrozenEMFunction` | 2..max(16,2*outputs) centers, positive spreads, p in [1,4], q in [1,2], output map. Softmax logits -log(2pi)/2-log(sigma)-(distance_p/sigma)^q; the constant has no input-dimension multiplier. |
| `FrozenProductFunction` | Two cheap children on identical inputs, coordinatewise product. Cheap means linear, quadratic, tree, discretization or RFF; no recursive product, MLP or EM child. |
| `FrozenMLPFunction` | 1..3 linears; shared hidden width 1..127 when present. Hidden activations plus optional start/end activations retain their actual stage order and calibrated constants. |

Discretization permits one center, covering degenerate source count cases. Its
distance comparison uses the direct power/sum/root equation when representable,
with power-of-two scaling for overflow or underflow. It compares mantissa/exponent
pairs so logarithmic rounding does not turn adjacent represented distances into
false ties. Opposite-sign
finite coordinates whose subtraction overflows are compared using half-scaled
differences. One-axis distances retain the absolute difference directly, avoiding
pow/root rounding of adjacent values. p below one is a quasi-distance. EM uses the corresponding logarithmic
distance for spread scaling. These float64 arithmetic adaptations do not promise
bit identity with the source implementation. Intermediate arithmetic that still
produces a nonfinite final output rejects; it is not silently clipped or retried.
Tree averaging rescales only if the ordinary sum overflows, preserving finite
means such as two leaves both equal to 1e308. This is another arithmetic adaptation.

MLP activations reuse `CalibratedActivation`: population standardization,
rescale/shift, primitive, population standardization. Their records retain sample
identity/digest and shift-row indices. Supplying these records alone does not prove
that they were fitted independently or on the actual preceding stage. Source
sampling and the ancestral fitter must establish those bindings separately.
Authored wrappers retain all supported primitive parameters and affine modifiers,
up to six named parameters. This does not expand the future source selector's law.

## Admission and JSON

`frozen_function_from_dict(data, limits=...)` reconstructs the same typed records
from `json.loads(json.dumps(dataclasses.asdict(record)))`. Before tuple coercion,
it bounds raw traversal, nesting, numeric entries and estimated storage; keys must
be strings and arrays must be lists/tuples. Unknown fields/versions, mapping or
string arrays, inappropriate child families and invalid geometry reject. Decoding
admits one evaluation row with center-dependent working floors. Evaluation admits
each actual row count, including nested child/activation work, before conversion
to float64 or creation of parameter tensors. Caller limits come from the caller,
not the payload. Constructing an authored Python record itself is not a memory
admission gate; its input tuples already exist.

`parameter_count` counts retained numeric coefficients and structural metadata;
the shared `CausalLimits` admission also includes conservative geometry overhead.
For a 2-by-2 linear map and one row, four retained coefficients become 12 admitted
parameters and 1,984 estimated bytes: 56 float64 working channels plus 128 bytes
per admitted parameter. Either one-below limit rejects. The estimates are not
measured RSS or hard process-memory bounds. Whole-graph admission of all plans,
ledgers, live ancestors and simultaneous fitting buffers remains required.

## Development verification

The public tests contain literal checks for linear orientation, nonsymmetric
quadratic cross terms and constant, strict tree split bits and averaging,
discretization order/ties, Fourier phase arithmetic, EM's scalar normalization,
same-input product children and both MLP activation positions. All eight records
round-trip through JSON and preserve evaluation across chunks, repeated calls and
mutation of returned tensors. Input tensors, records and Torch's global RNG state
remain unchanged. Exact/one-below limits cover parameters, estimated bytes, node
counts, maximum/total widths and actual evaluation rows.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_frozen_functions.py
```

Current focused result: **15 passed**. Development RED/GREEN and static-check logs
are retained locally under `/tmp/r0-28-functions-*`; the false-logarithmic-tie RED
remains part of the history. Independent review rejected initial source `690297b`:
per-center magnitude scaling broke exact L1 ties (expected first-center output 15,
observed 13). That failure is retained at `/tmp/r0-28-functions-spec-690297b/`.
The repair uses exact power-of-two scaling. An additional author RED/GREEN verifies
overflow-safe tree averaging. The initial full check was stopped after the review
rejection. Review of repair `c58706e` passed the original 98-case tie sweep, then
rejected an accepted MLP activation that could not round-trip through JSON. Its
decoder now accepts all six supported authored activation parameters; the source
selector remains separate. The second rejection is retained at
`/tmp/r0-28-functions-spec-c58706e/`. Both interrupted full checks remain incomplete;
review of `ed5b67a` then found a two-axis p=0.5 tie: (64,0) and (9,25) both have
distance 64, but scaling selected the second center. That rejection is retained
at `/tmp/r0-28-functions-spec-ed5b67a/`. The direct-equation preference repairs
this additional rounding error. Frozen source is
`cbbaff2cb4c8c6e485ad3814f79e84f3c1d68967`, compared with base
`cb7977a916ff6f11d95790b2de4dd591ab8d3f39`.
Full `make check` passes **868 tests, one skipped**, with all static gates;
pytest took 464.14 seconds. The command uses the same environment above with
`make check` replacing the focused invocation.
[Raw output](evidence/issue-28/legacy-research/0028-frozen-functions-check-cbbaff2.txt) and
[development archive](evidence/issue-28/legacy-research/0028-frozen-functions-development.json) retain
52 files, including intermediate RED/GREEN, tooling failures, both interrupted
checks and all standards reviews. Intermediate development snapshots are not all
retained and are not independent exact-revision proof. The archive's per-file
lengths/hashes were checked after packaging. Independent review uses separate
pinned source exports and retained procedures; its final report follows below.
The [CLI help](evidence/issue-28/legacy-research/0028-frozen-functions-cli-help.txt) still lacks train/eval:
conditional `make smoke` and `make eval-smoke` are NOT_AVAILABLE.
[Publication gates](evidence/issue-28/legacy-research/0028-frozen-functions-publication-gates.txt) pass
checkout/hooks, tracked-file/artifact limits, formatting, lint and file-token caps.
Publication preserves the frozen source and tests. Raw check output retains its
original whitespace.
CPU was exercised; CUDA execution is NOT_RUN.

## Independent component verification

The final specification verdict is **PASS_COMPONENT** at `cbbaff2`, with
**948 cases** across seven standalone programs:

| Evidence | Cases |
| --- | ---: |
| Independent equations, compositions, state and replay | 142 |
| Malformed JSON, geometry, input and explicit numerical rejection | 157 |
| Exact/one-below limits and aggregate admission | 291 |
| Cached-source transform comparisons with authored state | 258 |
| Original L1 ties, MLP JSON and two-axis quasi-distance reproducers | 100 |

The 142 equation cases each include JSON reconstruction, independent expected
outputs, chunking, repeated evaluation after mutating returned tensors and input,
record and global RNG preservation. All 25 cheap-child pairs and all 12 MLP
depth/activation-flag combinations occur. Admission cases independently derive
resource totals and check rejection before conversion/allocation. The 258 source
comparisons execute cached transform bodies only, with no source sampling/fitting;
their provenance is qualified, not freshly verified upstream.
[Report and exact commands](evidence/issue-28/legacy-research/0028-frozen-functions-spec-report.md),
[totals](evidence/issue-28/legacy-research/0028-frozen-functions-spec-totals.json).

Separately, **18 prior-r0 comparisons** are byte-equal between base and source:
three existing profiles, two graph families and three task types, with seed 649,
16 rows, eight context rows and two features. Both revisions completed all 18
cases. Public recipe decoding, generation, explanation and group save/verify/load
preserve generated records and saved archive member hashes. This is bounded
regression evidence, not a claim about every old profile. It is distinct from the
cached upstream transform comparisons.

The [initial standards review](evidence/issue-28/legacy-research/0028-frozen-functions-standards-690297b.txt)
found zero documented violations and one nonblocking repeated-dispatch advisory.
Three subsequent repair deltas found no new violations or advisories; the
[last delta](evidence/issue-28/legacy-research/0028-frozen-functions-standards-cbbaff2.txt) preserves that
scope distinction. Standards and numerical verdicts are separate.

Ten exact, formatted and executed verification/custody programs are published in
[scripts/verify_frozen_functions](../scripts/verify_frozen_functions).
[Archive metadata](evidence/issue-28/legacy-research/0028-frozen-functions-spec-artifacts.json) pins all
482 original file identities, the original manifest itself and 461 files from
the successful fresh replay. Source/base exports are reconstructed from Git.
Recovery verified all **944 publication-manifest entries**;
[receipt](evidence/issue-28/legacy-research/0028-frozen-functions-spec-recovery.json).

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-frozen-functions-spec-artifacts.json \
  /tmp/frozen-functions-verified --repo "$PWD"
bash /tmp/frozen-functions-verified/replay.sh
```

Recovery checks bytes; the replay driver creates a new directory and reruns the
behavioral programs using the declared existing environment. It preserves the
restored evidence. The retained fresh replay is additional execution evidence,
not additional coverage in the 948/18 counts. A packaging attempt exceeded the
single-file artifact cap; its output remains in the author archive and the final
archive is split into three bounded parts.

## Retained rejected revisions

The three independent rejections remain separate from subsequent repairs:

| Source | Defect | Retained package | Verified original file identities |
| --- | --- | --- | --- |
| `690297b` | L1 first-center tie | [Metadata](evidence/issue-28/legacy-research/0028-frozen-functions-reject-artifacts.json), [recovery](evidence/issue-28/legacy-research/0028-frozen-functions-reject-recovery.json) | 2,573 |
| `c58706e` | Accepted MLP activation rejected by JSON | [Metadata](evidence/issue-28/legacy-research/0028-frozen-functions-second-reject-artifacts.json), [recovery](evidence/issue-28/legacy-research/0028-frozen-functions-second-reject-recovery.json) | 368 |
| `ed5b67a` | Two-axis quasi-distance tie | [Metadata](evidence/issue-28/legacy-research/0028-frozen-functions-third-reject-artifacts.json), [recovery](evidence/issue-28/legacy-research/0028-frozen-functions-third-reject-recovery.json) | 389 |

Each archive retains the original review, fixtures, programs, logs and manifest.
Pinned Git objects reconstruct source/base exports; every original manifest entry
was verified before packaging and after recovery. The publication manifest also
checks the original `manifest.json` itself, so its entry count is one larger.
Recovery verifies artifacts; it is not a new behavioral execution.

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-frozen-functions-reject-artifacts.json \
  /tmp/frozen-functions-rejection --repo "$PWD"
```

Use a new destination and the corresponding metadata for the other two packages.
The retained shell commands name original absolute paths: preserve those outputs.
For a fresh reproduction, copy the programs and reconstruct the specified source
into a new working directory with the same relative layout; set `PYTHONPATH` to
that directory's `source/src`, then execute the copied Python programs there.
The original failures are expected on their rejected source revisions.
The third package also retains separately labeled verifier mistakes: insufficient
Decimal precision in a preliminary oracle, lint corrections, summary wording and
an accidentally formatted cache copy restored to its original hash. These do not
replace the product rejection or count as product failures.

## Remaining integration

All source function requests, duplicate selector slots, weight/matrix/activation
subtrees, raw random ordering and draw manifests remain required. Tree feature
masses/split rows, discretization center permutations, GP product/radial spectra,
EM anchors/spread draws and MLP stage-fit ancestry need sampled provenance beyond
these numerical records. Root bases, graph attachment, complete transactions,
profiles, moments, generated traces, saved workbench/Colab flows and the wider
paper-option inventory remain in the full scope. This component establishes no
scientific, supported-scale, browser, hosted or complete-family sampling claim.
