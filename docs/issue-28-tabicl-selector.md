# TabICL activation selector

`sample_tabicl_activation_pipelines` constructs one frozen `CausalNodePipeline`
per independently supplied layer calibration matrix. `calibration_ids` names
those matrices and `rng` owns all sampling. The returned pipelines attach to
functional causal nodes and serialize through the existing recipe and dataset
archive formats.

The source pool has 18 simple types: tanh, leaky ReLU, ELU, identity, SELU, SiLU,
ReLU, softplus, ReLU6, hard tanh, positive-at-zero sign, Gaussian bump,
exponential, square root of absolute value, unit-interval indicator, sine,
square and absolute value. Ten equally weighted slots select a Fourier function.
With `diverse=True`, half the probability chooses one shared type and half chooses
types independently for each layer. Sharing a type never shares its sampled
Fourier coefficients or random scale/shift parameters.

Options retain the source switches and explicit extensions:

- `random_functions=False` removes the ten Fourier slots.
- `scale=False` removes outer standardization and random scaling. Fourier still
  has its own input standardization.
- `diverse=False` always shares the selected type across layers.
- `individual=True` samples per-channel instead of shared Gaussian scale/shift.
- `profile="paper"` uses the paper's Fourier exponent distribution;
  `profile="pinned-code"` is the default. Their difference is described in the
  [Fourier contract](issue-28-tabicl-fourier.md).

A scaled simple activation has stages standardize, random scale/shift, activation.
A scaled Fourier activation has standardize, random scale/shift, standardize
again, Fourier. The inner statistics are fitted on the transformed independent
calibration values. The epsilon is retained at both stages; the second stage
cannot be silently canceled algebraically.

This implements the selection and composition laws in the
[pinned source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_activations.py#L211-L258).
It uses explicit independent float64 calibration and a caller-owned NumPy RNG,
not source first-batch fitting or upstream mixed Torch/NumPy bit identity. Type
choices are drawn before per-layer numerical parameters. Archives retain the
realized stages and parameters; this API does not introduce a separate sampler
provenance record. The supplied samples must describe each layer's input;
the helper does not propagate them through a whole network or claim source
architecture/default-prior reproduction.

Aggregate worst-case channel, parameter and workspace limits apply before fitting
or sampling. Any later construction failure rolls back all caller RNG draws.
Nonfinite evaluation fails explicitly. Numerical variation is preserved rather
than retrying until an attractive function appears.

Fourteen focused selector tests cover all 18 simple types and ten Fourier slots,
five-engine type-choice oracles, fresh parameters under shared selection, both
Fourier profiles, shared/per-channel scaling, independent NumPy composed output,
aggregate rejection, late-failure RNG rollback and generated dataset JSON/ZIP
replay with repeated-row traces. Combined with component regressions, 38 checks
passed. Whole-issue acceptance still requires the remaining option audit and
current-head full checks; this is not a training reproduction result.
