# TabICL mask scaling profiles

`NetworkSamplingRequest.mask_profile` now accepts `tabicl-individual` and
`tabicl-block`, mapping `I1.code.11` and `I1.code.12`. They follow the variance
scaling in [the pinned TabICL MLP initializer](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_mlp_scm.py#L199-L217).
The existing `generic`, `p1-individual` and `p1-block` profiles are unchanged.
These are initialization components, not a complete TabICL prior.

- `tabicl-individual`: the first weight matrix is unmasked. Later matrices use
  `p=min(dropout,0.99)`. The retained Gaussian matrix multiplier is
  `weight_scale / sqrt(1-p)` when scaling is enabled. This is distinct from the
  historical P1 expression `weight_scale / (1-p**scale_exponent)`.
- `tabicl-block`: sample an integer block count uniformly from 1 through
  `ceil(sqrt(min(input_width,output_width)))`. Floor-sized diagonal rectangles
  define the mask; remainder rows/columns stay zero. Scale by
  `weight_scale / sqrt(mask.mean())` when enabled. The generic `block_limit`
  does not control this source-specific count law.

For either profile, set `scale_exponent=0.5` to enable source scaling or `0.0`
to disable it. Other exponents reject. Select the matching sparse family,
Gaussian weights and `fan_in=False`; the generic fan-in factor must not be
silently multiplied into this source law. Biases use the retained uniform
`[-1/sqrt(input_width),1/sqrt(input_width))` law. A sampled bias-scale hyperprior
rejects because it would be unused.

```python
request = priors.NetworkSamplingRequest(
    5,
    (7, 3),
    families=("individual-sparse",),
    mask_profile="tabicl-individual",
    scale_exponent=0.5,
    fan_in=False,
)
network = priors.sample_network(request, named_rng=named_rng, raw_rng=raw_rng)
```

The existing shared constructor retains every choice, original Gaussian matrix,
mask draw, bias, multiplier and both RNG stream boundaries. Source formulas use
r0's existing named/raw NumPy streams; upstream Python/Torch bit identity is not
claimed. Degenerate valid draws are not filtered or resampled.

Independent checks cover two profiles, enabled/disabled scaling and all five RNG
engines, including first-layer behavior, the 0.99 cap, exact raw Bernoulli masks,
block geometry, realized keep fractions, uniform bias bounds, JSON replay and
numerical traces. Ten pre-change P1 archives across the same engines remain
byte-identical. Existing network and hyperprior checks run separately alongside
these new cases. Evidence, including the original missing-profile failures and a
corrected test error-message expectation, is ignored under
`docs/evidence/issue-28-tabicl-masks/`. Current-head full-suite and independent
review remain pending; this does not close other source-profile gaps.
