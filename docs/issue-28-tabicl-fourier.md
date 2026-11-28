# TabICL finite Fourier activations

The new activation component addresses the sine-sum laws in `I1.function.01` and
`I1.code.01`. It is distinct from existing RFF target generators and the newer I2
product/radial spectra. The references are [TabICL v1 Appendix B.1](https://arxiv.org/html/2502.05564v1#A2.SS1)
and [the pinned activation implementation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_activations.py#L86-L110).

Both profiles sample 256 frequencies `a_i ~ U[0,256)`, phases `b_i ~ U[0,2*pi)`,
and Gaussian coefficients `z_i`, then evaluate

```text
w_i = a_i ** (-exp(u))
f(x) = sum_i z_i * (w_i / ||w||_2) * sin(a_i*x + b_i)
```

The **paper** profile samples `u ~ U[0.7,3]`; the **pinned-code** profile samples
`u ~ U[log(0.7),log(3)]`. These are deliberately different versioned activation
kinds: `tabicl-paper-fourier-v1` and `tabicl-code-fourier-v1`.

```python
activation = priors.sample_tabicl_fourier_activation(
    rng=np.random.default_rng(28), profile="pinned-code",
)
```

The ordinary `CausalActivation` record retains the original frequency, phase,
Gaussian and exponent draws as named numeric parameters. Reopened tasks therefore
reconstruct the same coefficients without resampling. This uses an owned NumPy
stream and float64 generation, an r0 execution adaptation; it does not claim
upstream Torch/NumPy bit identity. Optional term counts 1–4096 are explicit r0
exploration settings; 256 remains the source reference. Caller resource limits
apply before sampling, and failed sampling preserves the caller RNG.

Evaluation uses stable log-space normalization of the positive frequency factors.
It accumulates one sine term at a time, avoiding a rows-by-channels-by-frequency
allocation. The same sampled function applies independently to each channel.
Zero frequency is an explicit failure because the negative power is undefined;
nonfinite inputs/outputs fail without retries or clipping. A zero/weak output remains
valid. Moment certificates recognize the finite sine sum as bounded.

The component works through ordinary network affine activations and causal-node
pipeline stages, including JSON/task-group save/reload and repeated-row numerical
traces. Independent NumPy calculations verify both profile equations and exact
stream advancement across five RNG engines. End-to-end cases verify network and
causal pipeline outputs, large task IDs and saved replay. Raw evidence, including
repaired import-cycle/annotation failures, is ignored under
`docs/evidence/issue-28-tabicl-fourier/`.

This is the frozen nonlinear function component, **not the full source activation
pipeline**. Source-specific standardization, Gaussian shift/lognormal rescaling,
shared-versus-independent layer-type selection and the tenfold Fourier selection
weight still need their complete source-profile composition. Existing generic
calibrated standardization uses population variance and epsilon 1e-4; do not claim
it equals the pinned code's sample standard deviation plus 1e-6. Training-ready r0
must fit independent calibration data and freeze the resulting constants, rather
than learning them from query rows. These remaining source-profile requirements
stay in #28. Full-suite and independent review of this new component remain pending.
