# Issue 28: named numerical sampling preferences

This checkpoint supplies a public numerical sampler and immutable, validated
sampling records. Categorical preferences and integration into generator profiles
remain required. It does not complete the source joint prior or issue #28.

## Source and scope

The reference is TabICL commit
[`9530bd65a7b806f6ad4f9f4335af745577e9d3cc`, `_base.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py),
particularly `GlobalSampler.numerical` and `randint`. The implementation is
independent; no upstream implementation was copied. The existing source-option
inventory remains authoritative for the wider required coverage.

The public entry point is `r0.modules.priors`. `NumericalParameter` declares a
name, finite ordered bounds, shape, sampling mode, linear/log scale, optional
boundary mass and integer conversion. `ParameterSampler` owns one explicitly
injected NumPy Generator. `freeze()` produces `ParameterPreferences` containing
`NumericalPreference` and `NumericalDraw` records. These contain the first request,
effective preference mode, Beta hyperparameters where applicable, every subsequent
request, raw unit-interval values, realized values and preference references.
`parameter_preferences_from_dict` reconstructs and validates the record.

No ambient NumPy/Python/Torch RNG is consulted. A frozen record describes realized
sampling decisions; it is not a saved RNG state and cannot resume an arbitrary
stream. Repeating requests from the same initial injected RNG state reproduces the
sampler's output in the same NumPy environment. Cross-version bitwise RNG identity
is not claimed.

## Numerical modes

- `local`: independent uniform unit values on every request.
- `global`: cache the first unit vector by name and reuse it. With the default
  `strict` policy, later requests must have the same bounds, shape, scale, integer
  conversion and boundary-mass setting. Equivalent explicit/global and redirected
  meta requests share the same cache. Rejection occurs without advancing the RNG.
- `meta`: draw one mean uniformly in [0,1) and one total concentration
  log-uniformly between 0.1 and 10000, then draw fresh Beta(mean * concentration,
  (1 - mean) * concentration) values. Repeated requests with the same name share
  these hyperparameters even when bounds or shape differ. Beta values may round
  to either endpoint. A zero sampled mean is an explicit invalid-Beta failure;
  the sampler does not silently retry or clip the mean.

A request whose mode is `meta` uses the sampler's `default_mode` (meta, local or
global), matching the source configuration fallback. Explicit local/global
requests keep their declared mode. Each name has separate effective-mode scopes.
Local/meta preferences retain the first request as provenance, but no cached unit
values. References and first-use order are validated on load; every preference
must have at least one retained realization.

`global_policy="quantile"` is an explicit r0 extension: retain the same unit vector
and remap it through each request's current bounds/transform. Batch shape must
still match. This supports common latent choices across width-dependent parameter
ranges without reusing an out-of-range final value. The source instead caches a
final value under the name without checking changed bounds, transforms or shape.
Strict validation and the quantile policy intentionally do not reproduce that
unsafe cache behavior. These modes are not claims of exact upstream RNG parity.

## Mapping and integer laws

The source boundary-mass transform is retained: `clip(1.2*u - 0.1, 0, 1)` before
interpolation. Endpoint atoms are part of the declared law, not repairs. Linear
interpolation is a convex combination of the declared binary64 endpoints. Exact
Decimal arithmetic avoids overflowing the difference between finite extremes;
the result is rounded to binary64 once. Log interpolation uses 80-digit Decimal
log/exp arithmetic and exact endpoint branches. This finite-range adaptation can
differ by last bits from upstream NumPy arithmetic. Rounding can map an interior
unit value to a floating-point endpoint; the output interval is therefore closed.

`integer="round"` rounds that binary64 result to an integer using ties to even,
as in source `use_int`. `integer="floor"` requires distinct ordered integer
bounds, floors the interpolated result and clamps to [low, high-1], as in source
`randint`. These are distinct laws. Python integers avoid the source NumPy signed
integer overflow behavior; endpoints are converted to binary64 before mapping.
The decoder requires actual integer realized values for integer modes.

The loader recomputes each transformed value and rejects mismatches. It checks
unit support, shape, effective scope, global reuse contracts, preference identity
and the first request. This is structural consistency verification, not evidence
that an authored unit value was randomly sampled from its claimed distribution.

## Admission and accounting

`PreferenceLimits` defaults to 10000 draws, 100000 total realized values and
1000000 retained numeric parameters. All limits are positive integers below 2^63.
A request shape has at most eight positive dimensions and 100000 entries; an empty
shape is a scalar (one entry). The sampler computes the next retained cost before
RNG consumption or vector allocation. Rejected requests leave the retained record
unchanged. The decoder admits raw record dimensions before copying vectors into
decoded tuples or recomputing numerical mappings.

`parameter_count` counts numeric record leaves, including duplicated provenance:
request bounds and shape dimensions; preference unit vectors and Beta mean/total;
draw preference indices, request fields, unit vectors and realized values. It
excludes strings, Boolean transform flags and limit configuration. It is a
retained-record cost, not a number of model training parameters. Example: one
global two-element draw costs 13 (five in the preference and eight in the draw).
Generator matrix admission does not yet include this record because no generator
profile is wired to it at this checkpoint.

## Delivery state

Targeted RED/GREEN checks cover global consistency, Beta/local scope, extreme
ranges and integer laws, explicit quantile reuse and admission before RNG use.
Full repository checks and independent review are pending for this checkpoint.
The eight earlier causal generator profiles and all their streams are unchanged;
this checkpoint only adds new public sampler functionality. Categorical shared
versus refreshed vectors, full source activation weighting, generator preference
bindings/admission and new profile integration remain required next work.
