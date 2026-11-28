# Frozen empirical rank and quantile maps

`fit_quantile_map(calibration, kind=..., tail_probability=...)` fits one numeric
column using only the explicit supplied calibration vector. It returns an
immutable `FittedQuantileMap` with unique sorted knots, multiplicity counts,
transform kind, tail clipping probability and SHA256 of the ordered little-endian
float64 calibration bytes. `fitted_quantile_map_from_dict` restores the complete
frozen map. Applying the map does not fit again, mutate calibration, access labels
or draw randomness.

This is a bounded r0 observation primitive, not a claim of matching a particular
paper's quantile implementation. The caller must select and record the intended
calibration scope. For context-fitted transforms, supply context rows only; do
not concatenate query rows. The primitive cannot infer provenance from an array
or a hash. The observation-task integration below retains context row identities and
recomputes the fit during archive validation.

For knot multiplicities `c_j`, cumulative count `C_j` and total count `n`, the
stored knot probability is `(C_j - c_j/2) / n`. Three semantics are supported:

- `rank`: at any value, return `(number strictly smaller + number smaller or
  equal) / (2n)`. Ties receive midranks. Between knots the empirical CDF is flat.
- `quantile-uniform`: linearly interpolate knot probabilities. Values strictly
  below/above the observed support map to 0/1. At the endpoints themselves they
  retain their midpoint probabilities. Thus there can be a jump immediately
  outside the fitted support. A constant column maps its exact value to 0.5.
- `quantile-normal`: apply the uniform map, clip probabilities to
  `[tail_probability, 1-tail_probability]`, then apply the standard Normal inverse
  CDF using `sqrt(2) * erfinv(2p-1)`. The default tail probability is `0.5/(n+1)`;
  explicit values must be at least float64 epsilon and strictly below 0.5.
  Clipping is recorded rather than silently imposed. Other map kinds require zero
  tail clipping probability.

For calibration `[0,0,2,4]` and values `[-1,0,1,2,3,4,5]`, rank output is
`[0,.25,.5,.625,.75,.875,1]`; uniform output is
`[0,.25,.4375,.625,.75,.875,1]`. Normal output applies the inverse CDF to those
uniform values clipped to `[.1,.9]`. The saved test compares that result against
Python's independent `statistics.NormalDist.inv_cdf` implementation.

Both calibration and evaluation vectors require 1..20000 finite float64 values.
Knots must be strictly increasing with positive integer multiplicities, totaling
at most 20000. JSON decoding rejects extra fields and malformed records. Float64
interpolation across opposite extreme endpoints halves endpoints when the raw
interval width overflows; it does not silently collapse the midpoint to an
endpoint. Nonfinite inputs fail. The observation wrapper adds schema checks, explicit context-scope receipts and
a bounded parameter/working-array admission model.

Focused evidence covers ties, manual interpolation, independent Normal quantiles,
round-trip identity, no refit on extreme query values, singleton/constant columns,
opposite maximum float64 endpoints and invalid fitted records. Evidence remains
in ignored `docs/evidence/issue-30-coverage/quantile-*`. The integrated focused checks below cover saved observation tasks; full
current-head verification remains a separate requirement.


## Observation-task integration

`QuantileColumnSpec(column, kind, tail_probability=None)` selects a continuous
source column. Pass a tuple through `ObservationRequest.quantiles` or the
`quantiles` argument of `observe_missingness`. JSON requests use the same named
fields. The pipeline order is raw source → context-fitted quantile maps → ordered
measurements → typed encodings → missingness. Count/category/binary/ordinal source
columns are rejected for quantile fitting; explicit typed encodings can follow
the continuous transform. Targets are unchanged and never used to fit the map.

A v4 observation receipt retains the exact context row indices, specifications
and fitted maps. Original raw inputs remain privileged truth. Loading/replay
refits from those raw context rows, compares the complete retained quantile
receipt, then replays later transformations and masks. Changing context calibration
values or the fitted map invalidates replay. Changing query values does not change
the fitted map, although it may change transformed outputs and mask probabilities.
Restoring the source recovers original values and schema for separate structural
replay.

Selected columns and context indices must be unique and in range. Inputs are
CPU float64, finite, at most 20000 rows × 256 columns. Before fitting, the wrapper
admits twice the selected calibration-value count against `limits.parameters`
and an explicit array/fit storage estimate against `limits.working_bytes`.
These are admission estimates, not a universal measured Python peak-memory bound.
Requests with no quantile maps retain v1–v3 receipts and omit the new empty field
from run headers and coverage manifests, preserving existing resume identities.

[`quantile-tree-ensemble-cpu.json`](../configs/generation/quantile-tree-ensemble-cpu.json)
combines normal quantiles with affine/noisy/clipped measurements, ordinal encoding
and self-dependent missingness. A two-task CLI pilot saved successfully. A separate
NumPy/NormalDist oracle checks fitted values, encoded outputs, mask draws and
probabilities against the saved archives. Sixteen focused observation checks
cover context-only fitting, replay, source restore, resource/schema rejection,
combined encoding/masking, guarded resume and prior observation versions. The
older four-shard notebook pilot also resumed with identical manifest bytes and
zero new tasks. Raw evidence lives under ignored
`docs/evidence/issue-30-quantile-integration/`.
