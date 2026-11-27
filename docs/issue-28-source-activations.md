# Issue 28: source activation selectors and frozen calibration

This increment adds the pinned source's hierarchical primitive-selection laws and
an independently calibrated standard activation pipeline. Public authored causal
nodes can retain, generate, save, replay and inspect that pipeline. It does not yet
connect these decisions to the sampled causal/matrix profiles, sample their shared
`rr_scale` parameter, or add the remaining L2/Kumaraswamy wrappers. Those integration
steps, wider mechanisms, interventions and actual browser/Colab delivery remain
required by #28. The full goal remains active.

## Source and declared adaptations

Reference: TabICL `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`,
[`_activation.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py).
The implementation is independently written; source code was not copied.

The standard selector has four separately weighted fixed-family slots, ReLU power
and signed power. The plain selector has eight fixed-family slots plus ReLU,
signed, reciprocal-absolute and integer powers. Repeated slots retain separate
identity/mass: collapsing them into one category would change the hyperprior.
Both selectors share the `acts_torch` fixed-catalog preference, in source order:

`tanh`, `leaky_relu`, `elu`, `identity`, `selu`, `silu`, `relu`, `softplus`, `relu6`,
`hardtanh`, `sign_positive`, `heaviside_positive`, `gaussian_bump`, `exp`,
`interval_abs_le_1`, `sin`, `square`, `abs`, `log_clip_abs`, `rank`, `sigmoid`,
`round`, `modulo`, `softmax`, `tied_max`, `argsort`, `log_sigmoid`.

The four power parameter names follow the source. Positive real magnitudes use
log bounds 0.1..10; the reciprocal descriptor stores the positive magnitude and
applies the negative exponent to `abs(x)+0.001`. Integer exponents use the distinct
high-exclusive floor law 2..6, giving 2, 3, 4 or 5. Earlier r0 exploration profiles
retain their original sampling laws.

The numerical/categorical machinery uses injected NumPy randomness, binary64
arithmetic, retained Beta/polynomial preferences and explicit shared/refresh
categorical policy described in [shared matrix preferences](issue-28-shared-matrix-preferences.md).
This does not claim source RNG identity. Stable channel tie ordering, stabilized
log/power arithmetic and logaddexp softplus retain r0's declared primitive
adaptations; they are not bitwise Torch source parity.

## Public selection records

`SourceActivationSampler(rng, mode="meta", policy="shared")` owns one numerical
scope and one categorical scope. `sample("plain")` or `sample("standard")` returns
a primitive `CausalActivation`; standard here identifies the **selector**, not an
already fitted wrapper. `freeze()` returns `SourceActivationSelections`, containing
all selections, both preference records and exact draw references. Each selection
binds a top-level slot and either a fixed primitive or an exponent. The decoder
`source_activation_selections_from_dict` validates canonical requests, actual
primitive values, complete coverage and ordered first-use references. Dropped,
reordered, duplicated, orphaned or altered decisions cannot silently change the
saved selection law.

Local/global/meta defaults and shared/refresh categorical policies remain explicit.
Global numerical reuse is strict because each named exponent has fixed bounds and
shape. Numerical and categorical `PreferenceLimits` are separate; categorical
limits include nested weight hyperpriors. `parameter_count` includes both scopes,
selection references and duplicated retained activation parameter values. Consumers
must admit that combined count when integrating selections into recipes.

A selection runs against a copied RNG and forked scope bookkeeping. Frozen records
are shared, while mutable lists/maps are copied. The external RNG and sampler state
commit together only after the complete hierarchical decision succeeds. A rejected
child draw leaves both unchanged. Each child request still admits its own cost
before random sampling or allocation. No rejected generated task is automatically
retried or replaced.

## Frozen standard activation

`calibrate_source_activation(primitive, independent_sample, rng=..., scale=...,
calibration_id=...)` returns `CalibratedActivation`. The supplied floating sample is
rows by channels. Callers own its independent stream identity; this function has no
context/query/target inputs. Its resulting constants never refit during generation.

For each channel, fitting performs:

1. Population standardization with mean and inverse `(std + 1e-4)^-1`.
2. One independently sampled calibration-row index per channel. Its negative
   standardized value is the channel shift. Apply `scale * (x + shift)` with a
   supplied scalar scale in [1,10].
3. Apply the retained primitive, including channel-coupled primitives over columns.
4. Fit another population standardization with epsilon 1e-4 on those outputs.

The source fits on its input batch. Accepted decision 12 instead requires an
independent reproducible calibration sample. This explicit adaptation freezes all
constants before task generation. A calibration identity and SHA-256 of shape plus
little-endian float64 sample bytes identify the fitted sample; the sample itself is
not duplicated in the mechanism. This API cannot establish that a caller-provided
sample was independent merely from its label. Profile integration must supply the
separate reproducible stream and retain its provenance.

Constant calibration channels remain valid: inverse standard deviation is 10000.
Statistics subtract a represented reference value before scaling deviations,
preserving small spreads around large offsets. Opposite-sign subtraction overflow
uses magnitude-scaled original values for those wide-spread channels. Evaluation handles subtraction overflow when multiplication by
the inverse standard deviation yields a representable result. Genuine nonfinite
stage results raise explicitly; no task is clipped, filtered or resampled.

`CalibratedActivation` subclasses the primitive descriptor and retains its kind and
parameters alongside five channel vectors, sampled row indices, scalar scale and
calibration row count. All widths and finite constants are validated. Numeric
storage costs include these additional fields. Authored activated and stochastic
nodes include the fitted buffers in conservative resource admission. The fit admits
dimensions before converting its sample to tensors or consuming randomness. The
sample already supplied by the caller is outside that allocation boundary.

Shared recipe decoding, generation and original-row inspection all use the frozen
record. Diagnostics inspect the actual standardized/rescaled argument of the base
primitive. Existing sampled activation priors reject fitted wrappers unless a new
construction explicitly defines that law; calibration cannot be smuggled into an
old profile by retaining only its primitive name.

## Development evidence and remaining verification

Public tests use the pre-agreed sampling, JSON decoding, generation and trace seams.
The local plain-selector fixture checks known uniform units against the separate
slot and primitive catalogs. A worked identity pipeline on the independent sample
{-1,+1}, scale 2, reduces algebraically to `2*x/(2 + 0.0001*1.0001)`. Generation,
JSON replay and row inspection agree for 16 and 32 task rows without refitting.
A child-draw limit case verifies a rejected selection leaves the injected RNG and
frozen record unchanged. [Development RED/GREEN logs](research/0028-source-activation-development.json)
retain their exact outputs and hashes as working-tree evidence, separately from
frozen-source full checks. The offset repair has separate
[RED](research/0028-source-activation-offset-red.txt) and
[GREEN](research/0028-source-activation-offset-green.txt) logs. Final full checks at `d9193b8` pass **653 tests, one skipped**, including all static
gates. Independent repaired-source review passes the bounded checks below; no whole-issue
or scientific acceptance is claimed.


## Initial independent rejection and repair

The initial source `c7cf429` passes full repository checks (**652 tests, one
skipped**, all static gates) but is **REJECT** under independent numerical review.
The [retained report](research/0028-source-activation-spec-reject.txt) and
[raw fixture/oracle evidence](research/0028-source-activation-reject-artifacts.json)
show that magnitude scaling before centering distorted the standard deviation of
`[1e16, 1e16+4]`. Its exact population standard deviation is 2, but the original
inverse scale was about 9.93% wrong. Saved generation and trace replay reproduced
the wrong constants; their agreement did not establish mathematical correctness.

The repair centers on a represented calibration value before scaling deviations,
with an overflow branch for opposite-sign extremes. The regression retains the
independent Decimal oracle's constants and tanh query output. Final [full repair checks](research/0028-source-activation-check-d9193b8.txt) pass
653 tests, one skipped, including all static gates. The independent dynamic audit
is retained separately below. The initial rejection is
not erased by the passing original repository suite.


## Final independent verification and reproduction

The [final spec review](research/0028-source-activation-spec-repair.txt) reports
**PASS** for the bounded selector/authored-wrapper increment at `d9193b8`:

- Ten independent Decimal calibration fixtures, including original/negative large
  offsets, adjacent maximum floats, opposite-sign/asymmetric extremes, constant
  maximum/zero/subnormal channels, subnormal spread and ordinary values. Maximum
  inverse-scale relative error is 1.12e-16 and maximum fitted-function absolute
  error is 4.45e-16 within these fixtures.
- Four NumPy channel-coupled function oracles (softmax, tied maximum, rank, argsort).
- 480 independently predicted local selections covering all eighteen selector
  slots and integer outputs 2..5; six local/global/meta by shared/refresh policy
  combinations and explicit global exponent reuse.
- Nine malformed selection and six malformed calibration records rejected;
  legacy priors reject fitted wrappers. Both sampler forks isolate state; rejected
  numerical/categorical children preserve RNG and scope. Calibration admission
  rejects before consuming randomness.
- **72 exact legacy regression cases**: twelve existing profiles, three topologies,
  two seeds. Recipes, arrays, identities, mechanism JSON and row traces match
  `97ff24d` byte-for-byte; all 72 generated successfully. These cases do not cover
  classification. Two additional new calibrated binary/multiclass fixtures check
  analytic scores, JSON replay and original-row target traces.

[Receipt](research/0028-source-activation-repair-receipt.json) binds four final
executed programs and confirms 66 imported source-file hashes match the frozen
revision. [Raw evidence manifest](research/0028-source-activation-repair-artifacts.json)
retains the exact 1,955,447-byte ZIP in three ordered binary parts. Concatenate with
no separator and verify the manifest's SHA-256 before extracting into a new folder.
The archive contains all per-case raw outputs, fixtures, saved recipes/arrays/traces,
complete receipt and original harness errors; the baseline source checkout is
excluded and can be reconstructed from its Git revision. The overstrict oracle
mean tolerance, a stdlib module-name collision and a missing shell Python alias
remain harness history, distinct from the original product rejection.

The four published `scripts/verify_source_activation_*.py` files are byte-identical
to the final executed programs, with their publication names mapped in the manifest.
Use explicit pinned source and a fresh output directory, for example:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/d9193b8/src /path/to/python \
  scripts/verify_source_activation_numerical.py /new/numerical-output
```

Run selectors and coupled programs the same way. Run compatibility separately with
the base and repaired source trees and compare their `compatibility.json` records.
The record of that comparison is [retained here](research/0028-source-activation-compatibility.json).

[Initial standards review](research/0028-source-activation-standards.txt) found no
documented violations and one nonblocking ownership/duplication heuristic in
activation working-channel accounting. Its [repair delta](research/0028-source-activation-standards-delta.txt)
found no new findings and verified original evidence integrity. These reports did
not rerun the suite or replace mathematical verification.

The publication changes documentation, retained evidence and portable programs;
source/test bytes remain those checked and independently reviewed at `d9193b8`.
The one full-suite skip is the existing CUDA case. Live CLI help exposes only
replay/oracle/diagnose-priors, so smoke/eval-smoke remain NOT_AVAILABLE. Browser,
Colab, GPU, supported-size and scientific acceptance remain outstanding. Timing
and process peak RSS are NOT_MEASURED for this increment.
