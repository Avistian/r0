# Direct GP kernel recipes

`GaussianProcessRecipe` uses the common `generate_task(recipe, seed=..., task_id=...)`
interface and guarded generation runner. This is a bounded reference generator:
at most 1,024 task rows and 512 calibration rows. CPU factorization is cubic in
joint support size. It is not the bulk path for 20,000-row tasks.

```python
from r0.modules.priors import GaussianProcessRecipe, SpectralKernel, generate_task

recipe = GaussianProcessRecipe(
    kernels=(
        SpectralKernel(frequencies=(0.1, 0.2, 0.3, 0.4), weight=3),
        SpectralKernel(frequencies=(1.0, 2.0, 3.0, 4.0), length_scale=0.5, weight=1),
    )
)
task = generate_task(recipe, seed=27, task_id=0)
episode = task.episode()
```

All kernels take `axes`, positions in the randomly selected active feature list;
empty means all active features. `feature_scales` multiply `length_scale` on
those axes; empty means ones. The components must jointly cover every active
feature. `combine="sum"` normalizes positive component weights to sum to one;
`combine="product"` requires unit weights. There are 1–64 components. These are
covariance combinations within one GP draw, not products of sampled functions.

For differences `delta_d = x_d - y_d`, effective lengths `l_d` and
`r² = sum_d (delta_d / l_d)²`:

| Public record | Covariance before component weight | Extra controls |
| --- | --- | --- |
| `StationaryKernel` | RBF or Matérn 1/2, 3/2, 5/2 | `kind` |
| `RationalQuadraticKernel` | `(1 + r² / (2*alpha)) ** (-alpha)` | finite positive `alpha` |
| `SpectralKernel` | `exp(-r²/2) * product_d cos(2*pi*frequency_d*delta_d)` | nonnegative `frequencies`, one per axis; empty means zero |

The rational-quadratic construction and RBF limit as alpha grows follow
[GPML equation 4.19](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
A sum of spectral components implements the product-of-cosines construction in
[Wilson and Adams, equation 12](https://proceedings.mlr.press/v28/wilson13.pdf).
Frequencies are cycles per input unit; the paper's spectral variance on axis d
is `1 / (4*pi²*l_d²)`. Lengths control the envelope separately from frequency.
Zero frequencies reduce to RBF. Nonzero frequencies can give negative covariance.
Parameters are specified in the recipe; this generator does not fit them to data.

Cards retain effective lengths, frequencies and units, alpha, normalized component
weights, the joint support, sampled normal variates, calibration statistics and
factorization error bounds. JSON/TOML decoding preserves the selected record type.
Existing RBF/Matérn recipes retain their serialized fields and random streams.

Finite positive lengths and weights are required. Unrepresentable effective lengths,
normalized weights or rational-quadratic distances fail explicitly. Rational-quadratic
evaluation uses logarithms to avoid squared-distance overflow and preserve the large
alpha limit. Spectral evaluation reduces cycles modulo one before cosine and rejects
absolute phase differences above 2^40 cycles (a declared numerical limit, about
1/4096-cycle float64 spacing at its boundary), or nonfinite phases. Kernels never
silently lower a frequency, remove a component, add jitter or switch families.
The shared factorization also rejects indefinite covariance beyond roundoff tolerance
and calibration variation below that tolerance. Estimated calibration changes the
marginal distribution of the normalized scores; only the raw joint draw is Gaussian.

The independent walkthrough reads saved cards and reconstructs a scalar score using
a separate distance calculation and the elementary covariance formulas:

```sh
python3 scripts/run_check.py python scripts/exact_gp_walkthrough.py \
  --extended --out /tmp/r0-gp-kernel-evidence
```

It includes three alpha values, eight frequency bands, a mixed subset product and
maximum supported GP shape, plus the pre-existing intentional numerical failure.
It saves and resumes each case. These bounded cases do not establish bulk throughput,
hosted Colab interaction, GPU acceptance, or complete issue #27 coverage.
