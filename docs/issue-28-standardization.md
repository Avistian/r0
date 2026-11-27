# Issue 28: standalone frozen Standardize

`calibrate_standardization` fits one population-mean/standard-deviation stage on
a caller-supplied independent sample. Its immutable `CalibratedStandardization`
record applies through causal node input/output activations and ordered node
pipeline stages, including saved decoding, generation and original-row inspection.
This advances the source node pipeline's Standardize requirement. Independent
ancestral sampling and the remaining [accepted scope](specs/0028-causal-network-generators.md)
are still outstanding.

## Source equation and adaptation

The [pinned source-law audit](research/0028-node-pipeline-sampling-laws.md)
records population standard deviation with correction zero and epsilon outside
the square root:

```text
mean[j] = mean(calibration[:, j])
inverse_std[j] = 1 / (std(calibration[:, j], correction=0) + 1e-4)
standardize(x)[:, j] = (x[:, j] - mean[j]) * inverse_std[j]
```

This is one centering/scaling stage. The existing standard activation wrapper
remains a separate standardize/rescale/activate/standardize construction.
A constant channel or one-row fit has inverse scale 10000. Its fitted value maps
to zero; future values retain the same scale, rather than becoming identically
zero. No task, query or label rows enter this fitter implicitly.

The existing stable statistics/centering arithmetic is shared with the standard
activation fitter. It preserves small representable spreads around large offsets
and rescues intermediate subtraction overflow when the final scaled result is
representable. Genuine nonfinite input or output is an explicit failure. There is
no retry, replacement draw or clipping. Fitted values are stored as float64
constants; generation uses the existing float64 equation path.

Records carry kind `standardize`, per-channel mean and inverse scale, calibration
row count, caller sample identity, sample SHA256 and version
`independent-standardization-v1`. The digest binds the calibration shape and
little-endian float64 bytes. It is provenance, not proof that a caller chose an
independent sample. The source's first-use fit is adapted to explicit independent
fitting followed by fixed evaluation, as required by accepted decision 12.

## Placement and mathematical compatibility

The ordered stage list can place Standardize before the existing channel weights
and L2:

```text
node output activation → Standardize → channel importance → L2
  → converters → structural innovation → whole-node importance
```

The authored stage surface also permits deliberate alternative compositions.
The future source sampler must bind the source's feature-importance enable flag
to both Standardize and its sampled channel vector; this increment does not
silently add that sampling law. Direct node input/output placement is an explicit
r0 composition using the same frozen record.

Every new Standardize placement requires certified finite input second moments.
This is checked at recipe construction even when target normalization is off.
A finite fitted sample cannot certify a heavy-tailed population. A preceding
bounded activation can establish the certificate; a later zero channel multiplier
cannot justify fitting an infinite-variance input. For plain node equations
containing Standardize, the same fitted-normalization gates apply throughout the
equation, including upstream fitted activations/L2. Legacy equations without the
new stage retain their prior admission behavior.

Standardize is channelwise and preserves potential source ancestry. Its finite
affine transformation preserves the certified positive-moment order. Moment
cards are emitted for new Standardize equations, including graphs whose other
nodes use only ordinary Gaussian innovations. Target normalization retains its
independent calibration and target-readout semantics.

## Admission, replay and limits

The frozen numeric parameter count is two channel vectors plus one row count.
The existing fitter admission also reserves its generic bias/noise slots.
For the two-row/two-channel example, this is nine admitted parameters and 1920
estimated working bytes. The transform's temporary-buffer allowance is ten
channels per feature, matching the shared stable arithmetic. These estimates
are not measured RSS or hard operating-system memory limits.

Stage records pass through the public recipe decoder and saved ExplorerSession
workflow. Original-row traces retain the `standardize` kind, frozen parameters,
calibration identity and realized stage values. Context/query size changes and
replay do not refit the stage. Actual browser/Colab interaction remains a separate
delivery requirement.

Malformed widths, unsupported versions/kinds/shape parameters, nonfinite constants,
nonpositive inverse scales, empty/nonfloating fit matrices and exceeded declared
resource limits fail explicitly. A standalone Standardize record is not a
numerical converter mode: source converters remain identity, Kumaraswamy and
the complete standard activation wrapper.

## Component verification

Acceptance is **PENDING** until full repository checks and the independent
spec/standards reviews complete at a pinned source revision. GPU execution,
supported workload measurements, throughput and scientific comparisons are not
claimed by the authored surface. Source profiles, shared sampling choices,
independent ancestral fitting, and the full issue 28 deliverables remain in scope.
