# Ordered measurement maps and noise

`draw_measurements(stages, inputs, rng=..., limits=...)` applies an ordered tuple
of `MeasurementStage(kind, columns, parameters=())` to a finite float64 matrix.
Stages select columns in declared order and act on the preceding stage's values.
Other columns remain unchanged. The source matrix is not mutated. These laws are
explicit r0 extensions, not source/checkpoint reproductions.

| Kind | Law and default controls |
| --- | --- |
| `affine` | `scale*x + shift`, default scale 1, shift 0; signed and zero scales allowed. |
| `signed-power` | `sign(x)*abs(x)**power`, positive power, default 1. |
| `signed-log1p` | `sign(x)*log1p(abs(x))`. |
| `tanh`, `sin`, `exp` | Elementwise named function. |
| `clip` | Clamp to finite `[lower, upper]`, default [-3, 3]; equal endpoints allowed. |
| `round` | Nearest multiple of positive `step`, default 1, ties to even. |
| `gaussian-noise` | Add independent `scale*Normal(0,1)` per selected cell, scale default 1. |
| `shared-gaussian-noise` | Same law with one draw per row shared by selected columns. |
| `heteroscedastic-noise` | Add `(scale + slope*abs(current_x))*Normal(0,1)` per cell; nonnegative scale/slope default 1. This multiplier is standard deviation, not variance. |
| `outliers` | Add `Bernoulli(probability)*scale*Normal(0,1)` per cell; probability default .01, scale 10. Uniform membership draws precede a full matrix of normal draws. |
| `student-noise` | Add `scale*T(df)` per cell; positive df default 3, nonnegative scale default 1. |

Parameters are immutable `(name, value)` tuples. Unknown, duplicate and inactive
parameters fail. Noise scales must be nonnegative. Noise stages still draw at zero
scale or probability; a valid weak/constant/no-outlier realization is never repaired.
Shared-noise columns use the same row draw; other noise laws draw independently in
row-major selected-column order. The Student law does not assert finite variance
for every permitted df, and no variance normalization is hidden inside it.

Order is meaningful: clipping then adding noise can leave the clipping interval;
adding noise then clipping cannot. Only explicit clip stages clamp values. No
empirical fitting, context/query normalization or hidden rank calibration occurs.
A nonfinite result fails the whole pipeline and restores the caller's RNG, even
if previous stages consumed draws. Inputs remain unchanged on failure.

The kernel admits 1..64 stages, 1..20,000 rows and 1..256 columns against a declared
workspace limit. It returns final values, the ordered stages, initial/final NumPy
snapshots and a binding hash of stages, input shape and original input bytes.
It retains no full intermediate matrices; those can be replayed from the source
and initial RNG. The observation-task integration below now retains raw-value
custody and composes measurements before masking in API/CLI runs. General schema
changes and arbitrary interleaving of masking/transforms remain open. Fitted rank/quantile maps, censoring/discretization/encodings,
engineered/redundant features and further required measurement families remain open.

Checks compare deterministic maps with scalar arithmetic, all five noise laws
with independent draw oracles across five RNG engines, order-dependent clipping,
outlier endpoints and complete RNG rollback after a late overflow. These establish
the named laws, not downstream robustness or the whole #30 portfolio.


## Measurement-then-mask observation tasks

`observe_missingness(..., measurements=stages)` and
`ObservationRequest(..., measurements=stages)` apply ordered measurement stages
before the declared mask. JSON requests use a `measurements` array of objects
with `kind`, `columns` and optional `parameters` pair arrays. All stages act on
declared continuous columns; category, ordinal, binary and count schema changes
are rejected until a typed conversion is explicitly specified. Empty schemas
retain the existing all-continuous interpretation.

```sh
r0 generate --config configs/generation/measured-tree-ensemble-cpu.json --out /tmp/measured-tree-ensembles
```

This example samples a mixed tree ensemble, then scales/shifts, adds Gaussian
measurement noise and clips its continuous first column before self-value MNAR
masking. Labels and structural mechanisms remain those of the original source.
MAR/self-value masks use the measured values; target/latent policies retain their
explicit privileged drivers. To request measurements with no missing cells, use
an explicit MCAR plan with probability zero.

Measured observations use `r0-missing-observation-v2` receipts. These retain ordered
measurement stages, input binding and initial/final RNG snapshots alongside the
mask receipt. Measurement PCG64 draws use a distinct domain tag plus observation
seed and source seed/task ID; changing mask draws cannot consume measurement draws.
Raw original inputs remain in privileged truth. Replay reconstructs measured values,
then mask probabilities and draws, and compares all present observed cells. Even a
change hidden by clipping must agree with the original measurement input binding.
Source restoration returns the original raw generator task, not the measured table.

Empty measurement pipelines retain v1 mask-only receipts and existing run-header
layout. Existing mask-only CLI output can resume without regeneration. Tests check
independent affine/noise/clip arithmetic, conditional mask probabilities/draws,
raw-value tampering, source restoration, typed-column rejection, guarded JSON
transport and resume. Additional measurement families, learned/fitted calibration,
typed conversions and arbitrary ordered mask/transform compositions remain open.
