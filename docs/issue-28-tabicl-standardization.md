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
the complete source selector and the additional internal standardization of
Fourier/random-frequency sine still require integration.

The conventions follow the pinned source documented in
[the Fourier source contract](issue-28-tabicl-fourier.md). Scoped verification
covers independent NumPy statistics, five NumPy engines with shared/per-channel
Gaussian draw oracles, constant and extreme channels, caller RNG preservation,
limits, malformed records, width checks and pipeline archive replay. The current
49-test focused run also includes existing standardization and structural-network
regressions. It is not a full-suite or whole-paper acceptance result.
