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
distance comparison uses power-of-two scaled mantissa/exponent pairs so logarithmic rounding
does not turn adjacent large represented distances into false ties. Opposite-sign
finite coordinates whose subtraction overflows are compared using half-scaled
differences. p below one is a quasi-distance. EM uses the corresponding logarithmic
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

Current focused result: **12 passed**. Development RED/GREEN and static-check logs
are retained locally under `/tmp/r0-28-functions-*`; the false-logarithmic-tie RED
remains part of the history. Independent review rejected initial source `690297b`:
per-center magnitude scaling broke exact L1 ties (expected first-center output 15,
observed 13). That failure is retained at `/tmp/r0-28-functions-spec-690297b/`.
The repair uses exact power-of-two scaling. An additional author RED/GREEN verifies
overflow-safe tree averaging. The initial full check was stopped after the review
rejection; full checks and independent repair verification are pending.
CPU was exercised; CUDA execution is NOT_RUN.

## Remaining integration

All source function requests, duplicate selector slots, weight/matrix/activation
subtrees, raw random ordering and draw manifests remain required. Tree feature
masses/split rows, discretization center permutations, GP product/radial spectra,
EM anchors/spread draws and MLP stage-fit ancestry need sampled provenance beyond
these numerical records. Root bases, graph attachment, complete transactions,
profiles, moments, generated traces, saved workbench/Colab flows and the wider
paper-option inventory remain in the full scope. This component establishes no
scientific, supported-scale, browser, hosted or complete-family sampling claim.
