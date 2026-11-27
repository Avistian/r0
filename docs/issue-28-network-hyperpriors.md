# Source-profiled neural hyperpriors

This opt-in library layer realizes depth, hidden widths, weight scale and optional bias scale before constructing a frozen standalone network. It feeds the existing seven-family sampler; no weights are trained against labels. Existing explicit requests and sampled-network archives are unchanged.

## Two source laws, not one purported reproduction

[P1 v6, Table 5](https://arxiv.org/html/2207.01848v6#A5.SS4) independently draws mean and standard deviation log-uniformly over the same bounds, then samples a nonnegative truncated normal. The rounded value precedes the additive offset. Layer bounds are 1–6 with offset 2; width bounds 5–130 with offset 4; weight-scale bounds .01–10 with no offset.

[Pinned TabICL code](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_hp_sampling.py#L151-L170) instead draws standard deviation as the sampled mean times a log-uniform factor from .01 to 1. Its helper truncates the normal **value** at one million before rounding and offset. This is later source behavior, not proof of a historical checkpoint's prior. These are separate `p1-paper-tnlu-v1` and `tabicl-9530bd6-tnlu-v1` law identifiers.

The clean-room numerical realization uses Python `NormalDist.cdf/inv_cdf` and retained uniforms. It discloses nearest-even rounding: the paper specifies nearest integer but does not disclose tie behavior. It makes no bitwise equivalence claim to upstream SciPy. Collapsed CDF intervals, rounded endpoints, nonfinite values or an operational cap breach reject without clipping, retry or replacement.

## Public records and semantics

`NetworkScalarPrior` offers those two TNLU laws and explicitly generic fixed, uniform choice, integer-uniform (inclusive endpoints), uniform and log-uniform laws. Bounds, rounding, offset and choice values remain frozen. A `NetworkScalarLaw` distinguishes the resolved mean/scale from each `NetworkScalarValue`; the latter retains its uniform, unrounded value and final value. Exact raw draws also remain in the parent ledger.

`NetworkHyperpriorRequest` contains an ordinary `NetworkSamplingRequest` template, scalar priors, width scope, depth semantics and operational caps. The template supplies exactly one width: the required output width. Its weight scale is the placeholder 1; the sampled weight law explicitly replaces it. Bias scaling is optional; when absent the existing template law remains. P1 mask templates use uniform biases, so requesting a bias-scale realization there rejects rather than recording an unused draw.

- `hidden-blocks-plus-output`: a realized depth L produces L hidden blocks plus the fixed output block. This matches the **count** of affine stages in pinned P1 noncausal code, not its complete activation/noise ordering.
- `total-blocks`: the realized depth includes the output block. Zero hidden blocks consume no width realization.
- `shared`: one hidden width realization is reused throughout the network.
- `per-hidden-block`: one resolved width law produces independent realizations for each hidden block. This heterogeneous architecture is an r0 extension.

All seven existing dense/sparse/skip/residual/irregular/gated templates remain available. Source scalar laws composed with these templates are **law-level adaptations**, not claims that the papers used each architecture. Existing activation, mask, weight-distribution, fan-in and gate choices remain explicit in the template and effective request. The source helper does not silently replace them.

```python
prior = network_source_hyperprior(
    NetworkSamplingRequest(2, (1,), families=("dense",)),
    source="p1-paper",  # alternatively "tabicl-9530bd6"
)
network = sample_hyperprior_network(
    prior,
    hyper_rng=hyper_rng,
    named_rng=named_rng,
    raw_rng=raw_rng,
)
recipe = NetworkRecipe(task, network)
```

`SampledHyperpriorNetwork` is a versioned sampled-network envelope. It retains the effective old request/function/choices/raw records and child snapshots, plus the profile, scalar records, hyperprior raw ledger and hyperprior snapshots. Full replay re-resolves the laws, reconstructs the exact effective request and samples the existing architecture. Decode rejects altered source labels, parameters, raw values, effective dimensions or stream boundaries. Generation, input views and variance-matching custody checks invoke that full replay. Changing episode/query identities never resamples these construction parameters.

## Admission and owned randomness

Three distinct injected generators own hyperprior, named architecture and raw matrix/mask draws. The outer transaction commits only after the entire admitted realization succeeds. A child matrix/bias overflow also rolls back all three streams.

Default operational capacity is eight hidden blocks of width at most 32. These are engineering caps, not truncations of the source law. Larger explicit caps remain configurable subject to existing geometry, raw-draw and memory budgets. The maximum declared materialization envelope plus scalar metadata/snapshot overhead is admitted before scalar draws; realized sizes are checked before allocating layer/width structures. A source draw exceeding capacity is a reported failed attempt. Successful bounded draws must not be described as an unconditional sample of the unbounded paper law.

Author evidence retains a fixed 32-attempt, zero-retry structural census, source-law SciPy quantile checks across three engines, seven-family composition/replay, adversarial mutations and real overflow rollback. Nine old sampler hashes are compared with base2bd. This is structural verification, not predictive-quality or training evidence.

## Required follow-ons

The guarded `network-hyperprior` constructor now exposes 28 shared explorer/notebook presets: two scalar-law profiles × seven r0 architecture families × shared/per-hidden-block widths. Inspection displays realized dimensions/scales and full hyperprior custody; frozen Next and replay preserve them. The editor creates a linked authored frozen network while retaining the original sampled archive. Source-cap and timeout failures retain receipts and the active dataset. Public local/mock-notebook and actual Chromium checks cover construction, inspection, replay, exact identities, editing, reopening and export. No hosted/GPU execution or full-suite claim is made here.

P1.net.04–06 (input importance and shared/per-node internal noise) and P1.net.09 (noise-scale law with an actual runtime consumer) remain required. This change deliberately does not sample an unused noise/root-count parameter. Source-specific first-layer exemption and paper-versus-code noise/activation ordering need their own stochastic-network records and traces. Weighted mixed-choice hierarchies from pinned code also remain distinct from this template's explicit uniform catalog selection. No broader paper option is silently declared complete.
