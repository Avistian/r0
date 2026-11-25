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


## Spatially varying smoothness and amplitude

`GibbsKernel` uses the same axes, effective lengths and weights. Each extra tuple
has one entry per selected axis. Empty tuples select the defaults below:

| Field | Default | Effect |
| --- | --- | --- |
| `centers` | zeros | Position of each smooth transition |
| `transition_scales` | ones | Positive transition width in input units |
| `length_slopes` | zeros | Signed change in log length; magnitude at most log(1e6) per axis |
| `amplitude_slopes` | zeros | Signed change in log amplitude; total absolute magnitude at most log(1e6) |

For `t_d(x) = tanh((x_d - center_d)/transition_scale_d)`, local lengths are
`l_d(x) = effective_length_d * exp(length_slope_d * t_d(x))` and the amplitude
is `a(x) = exp(sum_d amplitude_slope_d * t_d(x))`. This gives smooth bounded
transitions, positive lengths and amplitude between 1e-6 and 1e6. Values are
specified in the recipe. They are not fitted from task observations.

The covariance is the product over selected axes of
`sqrt(2*l_d(x)*l_d(y)/(l_d(x)^2+l_d(y)^2)) * exp(-(x_d-y_d)^2/(l_d(x)^2+l_d(y)^2))`,
multiplied by `a(x)*a(y)`. This uses [GPML equation 4.32 and section 4.2.4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
The prefactor is required: substituting varying lengths into ordinary RBF alone
would not in general give a valid covariance. Zero slopes recover stationary RBF;
zero length slopes with nonzero amplitude slopes vary only amplitude.

Cards retain all transition parameters and the modulation formula. Numerical
limits are explicit: a local length that overflows or rounds to zero fails, as
does any nonfinite combined covariance. Products of many amplitude-modulated
kernels may exceed float64 range even when every component is valid individually.
Final overflow fails with the task identity; no component or strength is silently removed.
Products containing Gibbs kernels accumulate signed log covariance before exponentiating,
so shrinking and expanding envelopes can cancel without intermediate underflow.
Scaling lengths before squaring avoids intermediate overflow and underflow in the
Gibbs prefactor and distance calculation.

`--nonstationary` on the same walkthrough checks varying lengths, amplitude-only,
a subset product with a spectral component and the largest supported GP shape.
It retains an intentional amplitude-product overflow failure. The numerical oracle
uses the elementary formula directly, separately from production scaling and logs.
These options are bounded reference examples; hosted verification, exploration
recipes and visual inspection remain separate requirements.
