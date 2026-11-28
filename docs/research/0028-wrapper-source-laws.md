# Issue 28: L2 and Kumaraswamy source laws and integration boundary

Fresh primary-source inspection: 2026-11-27. Read-only checkout preflight:
`/tmp/r0-28`, HEAD `ffc23fbf4871856fdaa96b7fac1f47c4c1f8afaf`, only untracked
`.venv` at entry. This file records research, mathematical deductions and proposed
integration criteria; it is not implementation acceptance. No upstream code was
copied. Scope authority: [accepted decisions 10–13 and completion rule](../specs/0028-causal-network-generators.md).

## Exact source register

All I2 links below pin `soda-inria/tabicl` commit
`9530bd65a7b806f6ad4f9f4335af745577e9d3cc`:

- [A: activation laws, lines 185–217](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L185-L217).
- [B: sampler and first-call fitting](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py#L25-L75).
- [C: converter contract and numerical branches](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py#L46-L73).
- [N: node construction and evaluation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L17-L73).
- [G: graph evaluation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_graph_function.py#L32-L42).
- [D: separate fitting pass](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_dataset.py#L40-L67).
- [Q: configuration defaults and IID help](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py#L10-L39).
- [M: MITRA-v2, arXiv 2609.04540v1, 2026-09-03, §2.3](https://arxiv.org/html/2609.04540v1#S2.SS3).

Line anchors were corrected during pipeline integration using the exact pinned
raw files: web-extracted text removes blank lines and has different line numbers.
The [sampling follow-up](0028-node-pipeline-sampling-laws.md) retains raw-file
hashes and line provenance; the source revision and mathematical laws are unchanged.

## Laws and sharing

For a calibration matrix C with m rows and d channels, I2 L2 stores one scalar
`f = 1 / (sqrt(sum(C_ij²)/m) + 1e-8)` and applies `f*x`. It does not center,
normalize each row, or divide the sum by d. Kumaraswamy stores per-channel extrema
`lo_j, hi_j`, plus scalar a and b shared across channels; its law is
`u_j = clamp((x_j-lo_j)/(hi_j-lo_j+1e-30), 0, 1)` and
`K_j(x) = 1-(1-u_j**a)**b`. Both a and b use [0.2,5] with logarithmic mapping and
names `kumaraswamy_a`, `kumaraswamy_b`. [A]

The numerical sampler defaults to meta sampling: a name shares Beta-distribution
hyperparameters, not necessarily its realized draw. Local mode maps fresh uniform
draws; global mode caches a realized value. Consequently, calling the source's
default a/b laws simply “log-uniform” loses their hierarchical distribution.
`num_conv_mode` is the separate categorical selector name. L2 has no sampled
scalar parameter name: its factor comes from calibration. [B][C][A]

First invocation fits a transformer, subsequent invocations reuse it. A shared
context shares the sampler across components. This is first-call dependence, not
automatic recalibration on every evaluation. [B] (lines 172–197)

## Source calibration versus accepted decision 12

The dataset source uses `n_train+n_test` rows. Default `ensure_iid=False` means its
first emitted graph pass supplies fitting values. With `ensure_iid=True`, a prior
pass of that same size is discarded and the next pass supplies the table. The
source explicitly offers separate-pass fitting; do not claim independent fitting
is absent from I2. Defaults also enable node L2 and Kumaraswamy, disable activation
warping, and select historical numerical conversion. [D][Q]

For r0, require a reproducible independent ancestral calibration stream before
emitting context/query rows, or certified analytic quantities when available.
Retain fitted constants, calibration identity, row count, channel width, law
version and input digest. Different query values, query count, labels and replay
chunking must not refit the wrappers. Calibrate downstream wrappers using already
frozen upstream mechanisms and the correct propagated state. These are proposed
implementation criteria under [decision 12](../specs/0028-causal-network-generators.md),
not a claim that source records these artifacts. A separately named first-emitted-
batch reference may expose source limits; it must not be sold as training-ready.

Independent L2 calibration estimates a second moment, so finite-moment admission
must apply to the wrapper's actual inputs, not just the final target. A finite
sample from an infinite-variance distribution does not certify its population
second moment. Bounded transforms can change that compatibility. This follows
from the law and [decision 13](../specs/0028-causal-network-generators.md); the
[existing moment policy](../issue-28-exogenous-laws.md#moments-and-normalization)
provides a conservative starting point. Kumaraswamy extrema are finite-sample
calibration statistics and do not themselves require finite population variance.

## Node order and converter ownership

The pinned node reserves converter channels in addition to latent channels. It
evaluates roots/parent functions, optionally standardizes and applies channel
importance, optionally applies whole-node L2, then converts reserved slices and
extracts columns. Gaussian noise and whole-node scalar importance occur afterward.
Target-group metadata can disable numerical warping. Graph descendants consume
the node's returned state; emitted columns are collected separately. [N][G]

The converter interface returns `(propagated_state, extracted_column)`. Numerical
identity returns `(x,x)`. For warp K, corrected mode returns `(x,K(x))`; historical
mode returns `(K(x),x)`. One numerical converter requires one channel. [C]

**Important discrepancy: historical raw tuple versus actual graph aliasing.**
The historical converter's second value is the supplied tensor itself. The node
supplies a basic slice of its state, then replaces that same slice with the first
returned value before retaining the second value. Basic slices are views, so the
nominal raw readout shares storage with the overwritten state. Inference from
[C, lines 63–73][C] and [N, lines 61–63][N], supported by
[PyTorch 2.15 tensor-view semantics](https://docs.pytorch.org/docs/2.15/tensor_view.html):
the pinned graph's historical numerical branch exposes warped state **and warped
readout**, before later noise/importance. Corrected mode has raw propagated state
and separately computed warped readout. Thus the two graph branches still differ
for descendants, although the historical “raw readout” description is not an
exact account of the integrated pinned code.

The original local alias fixture's exact program and execution receipt were
NOT_RETAINED. A fresh [standalone program](../../scripts/verify_converter_alias.py)
and [execution receipt](../evidence/issue-28/legacy-research/0028-converter-alias.json) now reproduce the premise in
the existing environment, with script hash, Python/Torch versions and all three
variants. State `[[2],[4]]`, a basic column view, and replacement equal to one
quarter of that view yield historical-view state and readout `[[0.5],[1.0]]`
with shared storage. Copying the raw readout or using corrected readout-only
semantics produces the distinct values recorded in the receipt. This proves
the alias premise only; no upstream or r0 generator executes in this fixture.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /path/to/existing/python \
  scripts/verify_converter_alias.py /new/alias-receipt.json
```

Required naming/disposition proposal:

| Variant | Descendant state | Readout | Status/boundary |
|---|---|---|---|
| Historical converter tuple semantics | K(x) | copied raw x | Explicit reconstruction preserving the documented tuple intent; adapted graph behavior |
| Pinned historical graph semantics | K(x) | K(x) | Source-derived alias consequence; preserve by value in a named variant, without making accidental aliasing part of r0's API |
| Corrected readout-only | x | K(x) | `I2.measurement.01`, #30 policy; expose #28 interface |

The inventory currently labels `I2.converter.02` historical hidden-state warp and
`I2.measurement.01` corrected readout. The newly found alias discrepancy requires
a recorded variant/disposition under the specification's rule to preserve source
disagreements. Neither the generic primitive nor a historical tuple returning
copied raw x alone reproduces all pinned graph behavior. [C][N]
[Inventory](0028-source-option-inventory.md),
[required converter record](0028-source-options-06.json),
[measurement record](0028-source-options-07.json).

## M2 equivalence remains UNKNOWN

MITRA-v2 §2.3 places standardization, random feature importance and L2 treatment
on vector-node outputs, after its mechanism/residual aggregation description and
before final feature/target selection. It does not specify the L2 reduction axes,
epsilon, exact denominator, calibration population, zero convention or whether
the norm is fitted once. “Per-node” does not establish I2's batch scalar law or
per-row unit norm. [M]

The paper's linked [release](https://huggingface.co/autogluon/mitra-finetune/tree/main)
was inspected at a listing showing `b4701e8`; it is labeled inference/fine-tuning
code. No exact Hybrid-SCM L2 source was established in this bounded investigation;
this does not prove that no such source exists. `M2.mechanism.16` therefore needs
an explicitly named reconstruction (for example I2-style calibrated scalar L2)
with these unknowns retained, rather than an exact-M2 reproduction claim. The
paper-option completeness rule still requires resolution/disposition before full
coverage can be claimed. [M][accepted specification](../specs/0028-causal-network-generators.md)

## Mathematical boundary cases and proposed verification

These are deductions/proposed checks from the stated laws, not measured upstream
robustness claims. Use independent high-precision calculations for acceptance.

- L2 with calibration rows `(3,4)` and `(0,0)` has factor
  `1/(sqrt(12.5)+1e-8)`; the second output remains zero. This distinguishes it from
  row normalization and elementwise RMS. Constant nonzero calibration retains a
  nonzero direction; all-zero calibration yields factor `1e8` and zero calibration
  output. New query values still use that factor. [A]
- For Kumaraswamy a=b=1, the warp equals the clamped normalized argument. Equal
  calibration extrema produce zero at the calibrated constant. Values below that
  constant produce zero; values above it transition over the explicit `1e-30`
  denominator before reaching one in exact arithmetic. Do not silently replace
  this with constant-zero evaluation for every future input. [A]
- Finite extreme L2 inputs can overflow naive squaring although a scaled norm and
  final result are representable. Opposite-sign extreme extrema can overflow a
  naive Kumaraswamy range; closely spaced large values can lose their spread in a
  careless rescaling. Stable equivalent arithmetic should preserve the mathematical
  laws; genuine nonrepresentable results should be explicit failures. The
  `[0,1]` domain clamp belongs to the declared warp and should be inspectable.
  It must not hide NaNs or serve as general-purpose numerical repair. [A]
  [Decisions 3, 12–13](../specs/0028-causal-network-generators.md)
- With small positive a/b or arguments near endpoints, cancellation in
  `1-(1-u**a)**b` needs precision-aware verification; endpoints, saturated outputs
  and zero dispersion are valid mathematical results. Stable log/exponential
  identities are an implementation choice, not a change to parameter ranges or
  a license for additional clipping. [A]
  [Decision 3](../specs/0028-causal-network-generators.md)

## Minimum coherent increment

Proposed acceptance boundary, derived from [decisions 10–13, 18](../specs/0028-causal-network-generators.md):

1. Immutable L2 and Kumaraswamy records; frozen fit/apply interfaces; strict record
   validation; explicit domain and numerical-failure handling; retained scalar
   parameter draws with `kumaraswamy_a/b` sharing scope; width-aware resource costs.
2. Independent ancestral calibration at the actual wrapper placement, followed by
   shared generation, trace/inspection and replay integration. Source node order
   above is a named variant; fitting a wrapper at a convenient but different point
   must be separately labeled. L2 should be applicable to a whole vector node,
   rather than independently normalizing its observed columns. [N]
3. A converter boundary with explicit channel selection and independent
   propagated-state/readout values. Integrate the historical branch into a graph
   with a descendant and an observed column, preserving/dispositioning both
   historical tuple intent and pinned graph alias consequence. Provide the
   corrected readout-only interface for #30 without claiming its observation-policy
   integration is delivered. [C][N]
4. Retain worked calibration/application numbers, a descendant counterexample
   distinguishing all three converter variants, mutation/alias independence,
   zero/constant/extreme cases, round-trip records, chunk/query independence,
   shared/local/global parameter evidence and admitted/rejected resource examples.
   GPU, browser, Colab and broader family delivery stay PENDING until executed.

A primitive wrapper is useful and independently reviewable, but it cannot close
`I2.converter.02` without the state/readout graph behavior. `I2.activation.24` and
the M2 reconstruction also need placement and calibration evidence, not only a
function named L2. These are proposed completion checks, not implementation claims.

[A]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L185-L217
[B]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py#L25-L197
[C]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py#L46-L73
[N]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L17-L73
[G]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_graph_function.py#L32-L42
[D]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_dataset.py#L40-L67
[Q]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py#L10-L39
[M]: https://arxiv.org/html/2609.04540v1#S2.SS3
