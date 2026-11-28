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
and initial RNG. This is a kernel, not yet a complete observation-task envelope:
schema effects, raw-value custody, composition with missingness and CLI requests
remain to integrate. Fitted rank/quantile maps, censoring/discretization/encodings,
engineered/redundant features and further required measurement families remain open.

Checks compare deterministic maps with scalar arithmetic, all five noise laws
with independent draw oracles across five RNG engines, order-dependent clipping,
outlier endpoints and complete RNG rollback after a late overflow. These establish
the named laws, not downstream robustness or the whole #30 portfolio.
