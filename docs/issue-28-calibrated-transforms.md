# Issue 28: frozen L2 normalization and Kumaraswamy transforms

This increment implements reusable fitted transforms through authored causal
generation, JSON loading, diagnostics, original-row traces and saved explorer
sessions. It is a component of the required source normalization/converter work.
Sampled post-node pipelines, shared `kumaraswamy_a/b` selection, converter state
versus readout semantics, and actual browser/Colab delivery remain required.
A primitive wrapper alone does not complete `I2.converter.02` or all of #28.

## Source laws and boundaries

The reference is TabICL commit `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`,
[`L2Normalize` and `KumaraswamyWarping`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L185-L217).
The [primary-source investigation](research/0028-wrapper-source-laws.md) records
the complete node order, first-call fitting and optional source `ensure_iid`
calibration. Implementation is independently written; upstream code was not copied.

For a calibration sample with rows `i` and channels `j`, L2 normalization freezes
one scalar `factor = 1 / (sqrt(mean_i(sum_j(x_ij^2))) + 1e-8)`. It scales the entire
node vector by that factor. This is neither per-row normalization nor separate
per-channel standardization. Zero calibration energy gives factor `1e8` and remains
valid. MITRA-v2 also mentions node L2 normalization but leaves its exact axes,
epsilon and fitting procedure UNKNOWN; the I2 law is not claimed as exact M2 code.

The Kumaraswamy transform freezes per-channel minimum `m` and maximum `M`, then
applies `u = clip((x-m)/(M-m+1e-30), 0, 1)` and `1-(1-u^a)^b`. The positive shape
parameters are scalar and shared across channels. The source sampler uses separate
named log-range requests for `a` and `b` over 0.2..5. The authored API accepts any
finite positive shapes as an explicit extension; it does not claim to sample them
from those source preferences yet. Constant calibration channels remain valid.

The clamp is a declared part of this source function, not a silent numerical repair
or task filter. Diagnostics distinguish values outside the calibration range from
values whose normalized coordinates actually require domain clamping. With a range
comparable to `1e-30`, a value above the recorded maximum can still lie inside the
normalized domain. Flags never remove valid draws.

The historical converter's declared tuple and graph-level mutation disagree: a
returned raw view is overwritten when the graph writes back the warped state.
The research note retains this alias premise and distinguishes intended historical
tuple semantics, actual pinned graph semantics and corrected readout-only behavior.
Those variants still require separate converter integration and evidence; this
increment does not erase the contradiction or treat them as equivalent.

## Public records and fixed calibration

`calibrate_l2_normalization(sample, calibration_id=..., limits=...)` and
`calibrate_kumaraswamy(sample, a=..., b=..., calibration_id=..., limits=...)` return
`CalibratedTransform`. Samples are nonempty floating rows-by-channels matrices.
The fitter admits geometry before converting/copying the sample into tensors.
The record retains kind, shape parameters, factor or ranges, channel and row counts,
sample identity, SHA-256 and `independent-transform-v1`. The digest covers shape
plus little-endian float64 sample bytes. A transform requires the same channel
width when evaluated. Generation and inspection never refit its constants.

Accepted decision 12 requires independently sampled calibration. These authored
APIs accept no context/query/target inputs, but cannot establish sample independence
from a caller-provided label alone. Callers own the independent stream and retain
the sample when their evidence requires it. JSON loading validates the frozen
record; it cannot reconstruct an arbitrary caller-supplied sample from its digest.
Changing a valid authored constant constitutes a changed mechanism, not a refit.
An authored L2 factor also does not certify finite population second moments
for its caller's sample. Future sampled pipelines must check that requirement
at the actual L2 input before fitting; the existing target-normalization moment
gate is a separate check.

`CalibratedTransform` is a specialized activation descriptor accepted at input or
output stages of authored activated, stochastic and matrix causal nodes. Its
kind cannot be created as an ordinary primitive without the required calibration
metadata. Existing sampled source selectors continue to require their own bound
primitive/calibration records. No existing profile law or record gains new fields.

Retained numeric parameter cost is three for L2 (factor, channel count and row
count), and `2*channels+5` for Kumaraswamy (two ranges, `a/b`, retained factor and
counts). Evaluation adds four or sixteen working channels per input channel to
the conservative causal array estimate. Fitting separately admits its calibration
rows. These estimates are not measured process RSS or hard OS enforcement.

## Numerical adaptations

Evaluation and fitting are implemented for the shared float64 generation path;
cross-device/precision acceptance needs its own evidence. L2 fitting scales by the
largest sample magnitude before squaring, and forms the reciprocal without first
overflowing the true norm. A representable normalization factor is preserved even
when naive squared norms or the unnormalized norm overflow. Genuine nonfinite
input or output still fails explicitly; no task is retried or replaced.

The range warp evaluates in the logarithmic domain across the range and CDF
stages. It preserves small ratios that would underflow before a fractional power,
and retains epsilon's effect near the maximum even when direct addition would
round it away. These are mathematical-equation adaptations, **not bitwise parity
with upstream eager floating-point arithmetic**. For example, range `[0,1]`,
`x=1`, `a=1`, `b=0.2` gives approximately `0.999999`; eager addition of `1e-30`
to one can instead produce exactly one. Range `[0,1e308]` at the smallest positive
binary64 input, `a=0.2,b=1`, gives about `5.48e-127`, rather than losing the input
ratio to zero. The positive-shape extension also preserves subnormal exponent
effects before later scaling. The upper boundary retains the error-free
subtraction residual so a coordinate
just inside the domain cannot become an endpoint through rounding. Numerical
branch approximations omit terms below
binary64 relative precision; independent high-precision evidence is required.

The moment checker recognizes the range warp as bounded. Heavy-tailed noise
before it can therefore yield a bounded output, while noise added afterward keeps
its own moment restrictions. L2 is a fixed positive linear scale and does not
improve a distribution's moment order.

## Verification state

[Development evidence](research/0028-calibrated-transforms-development.json)
retains missing-API REDs, two representable-tail failures, their GREEN logs,
the initial twelve passing targeted tests including six saved task workflows,
and subsequent boundary/decoder REDs and eight passing transform regressions.
These are working-tree development logs, not immutable snapshots of every
intermediate source. Reproduce current behavior from a checkout of the repaired
source with:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/existing/venv \
python3 scripts/run_check.py pytest tests/priors/test_calibrated_transforms.py \
  tests/explorer/test_transform_session.py
```

The [original standards review](research/0028-calibrated-transforms-standards-original.txt)
identified missing retained evidence for the development checks and alias fixture.
The linked archive preserves the available raw logs; the research note explicitly
marks the original alias receipt NOT_RETAINED and links a fresh retained execution.
The [narrow standards delta](research/0028-calibrated-transforms-standards-delta.txt)
resolves both findings and verifies artifact custody. Its scope is the evidence
delta, not source correctness or overall acceptance.

## Independent numerical review and retained rejection

The [original review](research/0028-calibrated-transforms-spec-reject.txt) rejects
`bf3a4c535bb87bee5a4557a1e0c30f70a29fb82a`: subtraction rounded an interior point
to the upper boundary. With calibration `[0, smallest binary64]`, `x=1e-30`,
`a=smallest binary64`, `b=1e-308`, application returned `1.0`; the independent
1,000-digit Decimal equation gives `1.419802591052941e-305`. The first run stopped
at case 527 after 526 passes. The original and final frozen scripts reproduce
that rejection. The interrupted full check on that source is explicitly NOT_PASS.

The [repaired review](research/0028-calibrated-transforms-spec-review.txt) gives
scoped PASS at `be0c793a043585b96b1825e659a10fd1f5deb7d1`:

- **874 original numerical cases**, with unchanged equations, grid and tolerances,
  plus **589 boundary/width cases** using 1,000-digit Decimal. These cover
  epsilon neighborhoods, signed subnormals, extreme/constant ranges, tiny/huge
  shapes, exact clamp diagnostics, L2 widths 1–1024 and frozen/chunked application.
- **172 integration checks**, including 36 authored compositions across both
  placements, activated/stochastic/matrix nodes and all three task types. Two
  additional dimension-changing dual-transform compositions pass independently
  calculated targets and original-row traces.
- **Six local saved/reopen/replay workflows**, frozen JSON validation, malformed
  records, exact/minus-one parameter and working-byte limits, row-count invariance
  and heavy-tail target-moment certificates. These are backend evidence.
- **168 unchanged legacy proposals** across 28 profiles, three topology families
  and two task ids: 164 successes and four identical explicit failures. Complete
  recipe, task, trace and status bytes match `ffc23fb`; no failure was replaced.

The initial invalid-record harness expected ValueError/TypeError but received
KeyError for an unknown calibration version. That fail-closed rejection was
accounted for explicitly; the original stop remains retained. The repaired decoder
now raises an intentional ValueError, covered by a separate regression. This
harness stop and the real numerical defect remain separate historical events.

[Raw evidence](research/0028-calibrated-transforms-spec-artifacts.json) preserves
original stops, complete inputs/outputs, old and repaired runs, saved sessions,
script/import hashes and custody. Six final programs under
[`scripts/verify_calibrated_transforms/`](../scripts/verify_calibrated_transforms/)
are byte-identical to executed copies; 96 imported modules were checked against
the repaired Git source. [Reproduction instructions](research/0028-calibrated-transforms-reproduction.txt)
identify the Python environment, stream/output controls and baseline comparison.
No upstream evaluator is used as the numerical oracle.

The archive uses ordered ZIP byte parts. Verify each part and the concatenated
archive SHA-256 before extraction. Five byte-identical legacy captures are stored
once with `duplicate-paths.json`; restore each `path` from its `copy_from`, then
verify every original file against the archive's `manifest.json`. This avoids
duplicating the same 46 MB payload while preserving every original byte and path.
The local original evidence is unchanged. Git source exports and caches are
excluded and reconstructable from pinned revisions.

Full `make check` at the repaired source passes **752 tests, one skipped**, in
373.95 seconds, with all static gates passing.
[Exact output](research/0028-calibrated-transforms-check-be0c793.txt).
The skip is the existing CUDA test on this CPU-only environment. Smoke/eval-smoke
remain NOT_AVAILABLE because the train/eval CLI commands have not landed.
Publication retains the checked source/test bytes and separately checks its
added scripts, documentation and artifact-size limits.

This component does not complete source sampling/shared a/b preferences,
converter/pipeline integration, the broader source inventory,
browser/Colab/GPU/scale evidence or the full issue.
