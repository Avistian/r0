# TabICL standardization and random scaling

`calibrate_standardization(sample, calibration_id=..., profile="tabicl")`
freezes channel means and inverse sample standard deviations, with correction 1
and epsilon 1e-6 outside the square root. It requires at least two calibration
rows. The serialized convention is `independent-tabicl-standardization-v1`.
The existing default retains population variance, epsilon 1e-4 and its original
version. Calibration uses an explicitly supplied independent sample; this is a
named adaptation of source first-batch calibration, not a claim of identical
training batches. Evaluation never updates the retained statistics.

`sample_tabicl_random_scale(channels, rng=..., individual=False)` samples
independent standard Gaussian draws a and b and freezes `exp(2*a) * (x+b)`.
By default all channels share one pair; `individual=True` samples one pair per
channel. The record retains the original Gaussian draws and declared width.
Construction checks limits before drawing and rolls back the caller's stream on
failure. Overflow is rejected rather than clipping the prior. Representable
results whose intermediate sum overflows use an equivalent scaled expression.

These stages compose in `CausalNodePipeline` and survive recipe JSON, dataset
ZIP save/reopen and numerical traces. Network stages check declared channel
width at construction. They are components of the source activation profile:
the [source selector](issue-28-tabicl-selector.md) composes these stages and
retains Fourier's additional internal standardization. The explicit sine factory
below also includes its own standardization.

The conventions follow the pinned source documented in
[the Fourier source contract](issue-28-tabicl-fourier.md). Scoped verification
covers independent NumPy statistics, five NumPy engines with shared/per-channel
Gaussian draw oracles, constant and extreme channels, caller RNG preservation,
limits, malformed records, width checks and pipeline archive replay. The current
49-test focused run also includes existing standardization and structural-network
regressions. It is not a full-suite or whole-paper acceptance result.

## Random-frequency sine

`sample_tabicl_sine(sample, calibration_id=..., rng=...)` returns two frozen
pipeline stages: TabICL standardization followed by sine with log-uniform
frequency on [0.1, 100] and uniform phase on [0, 2*pi). Positive ordered
`min_frequency` and `max_frequency` extend the default bounds. The frequency
and phase are stored as ordinary sine input scale/shift parameters, so existing
recipe decoding, generation, archive replay and tracing apply without a new
serialization format. Sampling uses a caller-owned NumPy stream; it does not
claim the same bits as the source Torch RNG. Invalid calibration, bounds or
parameter limits leave that stream unchanged.

This follows `RandomFreqSineActivation` in the
[pinned source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_activations.py).
Its independent calibration lifecycle is the adaptation described above.
It is an explicit option, not one of the source's 18 simple selector entries;
that list contains ordinary sine. The [complete selector composition](issue-28-tabicl-selector.md)
is available separately. Seven focused checks cover five-engine draw/numerical oracles,
construction rejection, JSON replay and generated dataset ZIP/tracing.
