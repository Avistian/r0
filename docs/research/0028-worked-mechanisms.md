# Independently worked mechanisms for issue #28

These are authored mathematical inputs and expected values. They do not come from
the generator implementation and are not claims that execution has passed. Rows
below are short mechanism traces; generated training episodes can repeat these
controlled inputs or use independently recorded longer draws.

## Scalar fork with a direct and mediated effect

Let `A = U_A`, `B = 2 A + U_B`, and `C = -A + 3 B + U_C`. Observe `A` and `B`,
predict `C`, and keep all three innovations in diagnostic truth only.

| Row | U_A | U_B | U_C | A | B | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.5 | -1 | 1 | 2.5 | 5.5 |
| 1 | -2 | 1 | 2 | -2 | -3 | -5 |
| 2 | 0 | -0.5 | 1 | 0 | -0.5 | -0.5 |

For example, row 0 has `B = 2*1 + 0.5 = 2.5` and
`C = -1 + 3*2.5 - 1 = 5.5`.

A hard intervention `do(B = 4)` removes the influence of `A` and `U_B` on `B`.
It preserves the original `A` and `U_C` in a matched comparison:

| Row | A after | B after | C after |
| --- | ---: | ---: | ---: |
| 0 | 1 | 4 | 10 |
| 1 | -2 | 4 | 16 |
| 2 | 0 | 4 | 13 |

A soft intervention changes only the B coefficient from `2` to `-1`:
`B' = -A + U_B`. Preserve the same innovations and the C mechanism.

| Row | B after | C after |
| --- | ---: | ---: |
| 0 | -0.5 | -3.5 |
| 1 | 3 | 13 |
| 2 | -0.5 | -0.5 |

Observational conditioning on `B = 4` does not mean either intervention. It does
not rewrite the B equation and does not generate these matched outcomes.

## Vector-valued node and channel intervention

Use row-vector convention. Let `A = [1, -2]`,
`B = A @ [[2, 1], [-1, 3]] + [0.5, -0.5]`, and `C = B_0 + 2 B_1`.
Then `B = [4.5, -5.5]` and `C = -6.5`. The weights map two parent channels to two
child channels. A hard intervention on `B_0` alone setting it to `4` preserves
`B_1 = -5.5` and produces `C = -7`. Observing only `B_0` hides a channel, not an
entire node; plots must show that distinction.

## Categorical outcomes versus categorical state

Logits `[0, log(3)]` give class probabilities `[1/4, 3/4]`. With inverse-CDF
sampling and classes `['class-0', 'class-1']`, uniforms `0.2`, `0.8`, and `0.25`
produce labels `0`, `1`, and `1` respectively (the boundary belongs to the next
interval). The full class vocabulary remains declared even if a particular sample
omits one class.

A native categorical causal variable is itself the sampled class, and its children
consume the recipe's declared category representation. A separate classification
adapter applied after a continuous target does not retroactively make that sampled
label the parent of existing descendants. The graph and export must identify which
construction is present.

## Valid constant and disconnected mechanisms

`X = U_X` and `Y = 7` form a valid task with a disconnected constant target. Raw
dispersion is exactly zero. If normalization is enabled, the recorded convention
must preserve a finite constant, retain the raw value and flag zero dispersion;
it must not divide by zero, discard the draw or resample its mechanism.

`Y = U_Y` with independent Gaussian `U_X` and `U_Y` is structurally disconnected
but can have nonzero output dispersion. Disconnectedness and zero dispersion are
different diagnostics. Neither is a hard generation failure.

## Paper/source activation differences

For inputs `[-1, -0.5, 0, 0.5, 1, 2]`, the paper interval indicator on `[0,1]`
gives `[0,0,1,1,1,0]`; the pinned source's `abs(x) <= 1` gives
`[1,1,1,1,1,0]`. Preserve two named variants.

For a channel vector `[2,2,1]`, a first-index one-hot argmax gives `[1,0,0]`,
while an is-maximum mask gives `[1,1,0]`. Ties distinguish these mechanisms.

## Calibration independence

For fixed independent calibration center `2` and scale `4`, raw values `[-2,2,6]`
normalize to `[-1,0,1]`. Changing query row count, query values, query labels or
context/query partition must not recompute these constants. A separately named
batch-dependent reference has different semantics and cannot be reported as this
fixed mechanism.
