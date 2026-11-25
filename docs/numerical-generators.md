# Finite weak signals and inactive inputs

Two corrections preserve valid tasks without changing sampled identities, retrying
failed draws, clipping inputs or substituting generator families.

## Nearly flat Fourier functions

The original Fourier evaluator uses `cos(x @ frequencies.T + phases)`. For very
large length scales its variation can be much smaller than its constant offset.
Rejecting all calibration scales below `1e-12` excluded representable functions;
merely lowering that cutoff would leave cancellation when subtracting nearby
cosine values.

If the original estimate is at most `1e-12` in any output channel, calibration and
signal evaluation instead use this identity, with `d = x @ frequencies.T`:

```text
cos(phase + d) - cos(phase) = -2 sin(d/2) sin(phase + d/2)
```

This is a constant shift of the same sampled function. Independent calibration
absorbs the shift, so the normalized mathematical function is unchanged. A scaled
standard-deviation calculation avoids squaring tiny raw magnitudes. No new random
values are sampled. The mechanism records `centered_basis: true`; the generation
graph exposes the actual projection and centered basis. The recorded center and
scale refer to this centered basis, not the old unshifted cosine values.

Recipes that previously succeeded retain their original calculation and card.
The `fourier-v1` decoder treats an absent flag as the original cosine basis. This
preserves existing replay while allowing previously rejected draws. Nonfinite or
numerically degenerate calibration still fails visibly. CPU evidence does not
establish precision or availability on every GPU.

## Inactive polynomial features

Inactive columns have zero coefficients, but raising them to a power first can
overflow: multiplying the resulting infinity by zero produces an invalid result.
Polynomial calibration and signal evaluation now compute powers only for active
columns. The full matrix shape and reduction order remain unchanged, preserving
ordinary successful results. The graph labels inactive basis columns as zero and
keeps the original unmodified input values visible in the input node.

This does not rescue overflowing active powers. Those invalid functions still
fail with their original task identity and a visible reason. The feature values
are neither clipped nor replaced in the training dataset.
