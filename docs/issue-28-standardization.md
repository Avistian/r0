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

## Option-level coverage

The standalone stage advances `M2.mechanism.14` (node standardization) and the
standardization required before the `I2.node.02` importance stage. Existing input/
output standardization inside the complete standard activation wrapper remains
separately recorded as `I2.activation.20` and `I2.activation.23`. The authored
operator does not close sampled source-node construction, its shared preferences,
or the complete source profile. The inventory remains 583 source/extension/
disposition records plus 140 specification anchors, not implementation counts.

## Component verification

Source: `962f0ada62a4eda3e71c45fa45bab0c4f66eb8fe`.
Baseline: `b7e755a0b1b505afcd8e2dd2e4ee9b85c9cf7261`.
Full `make check` passes **809 tests, one skipped**, all static gates; the test run
took 433.30 seconds. [Raw output](research/0028-standardization-check-962f0ad.txt).
The existing CUDA skip remains. The [CLI help](research/0028-standardization-cli-help.txt)
still has no train/eval commands, so smoke/eval-smoke are NOT_AVAILABLE.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

[Development records](research/0028-standardization-development.json) preserve
raw RED/GREEN and typechecking failures. Intermediate snapshots were not retained;
these are historical development evidence, not exact-revision acceptance claims.

## Standards

Independent review finds **zero hard violations** and one nonblocking possible
duplication: the exact-type activation admission list appears in both nodes and
pipelines. [Unedited review](research/0028-standardization-standards.txt).
No broader full-issue acceptance is implied by that scoped result.

## Spec

Independent scoped review finds **zero product defects**: 174 checks pass,
including 38 numerical comparisons and 58 expected rejections. Decimal/NumPy
oracles cover means, population scales, constant/one-row fits, large offsets and
representable intermediate overflow. Node placement, pipeline order, converter
branches, moment gates, provenance, decoding, and three saved task-type workflows
are checked through the public API. Maximum absolute error is 7.28e-12, within
predeclared relative tolerance 3e-14 and absolute tolerance 1e-14 (zero for inverse
scales); the largest absolute difference occurs on large-valued outputs.
[Unedited report](research/0028-standardization-spec-review.txt),
[totals](research/0028-standardization-spec-totals.json).

Independent [resource arithmetic](research/0028-standardization-resource-arithmetic.json)
checks three fitter sizes and direct input/output placements under both noise
orders. For example, the two-channel fitter admits nine parameters / 1920 estimated
bytes; the input-stage equation admits 13 / 9440 and output-stage equation 11 / 7744.
Each exact boundary admits and either limit one lower rejects. These remain array
estimates, not measured RSS or operating-system limits.

All 168 legacy proposals match the baseline byte for byte: 164 completed tasks and
four unchanged explicit numerical errors. Canonical record SHA256:
`03a5e4bd1334a6d0721b342491a8bcf124bd5bc4998c41395265608cbcd67969`.
Five final programs match their executed bytes; custody checks 80/80/100 modules
for the numerical/recipe/saved-workflow runs and 80/80 for the legacy revisions
against pinned Git. Two verifier mistakes remain retained as HARNESS_ERROR:
JSON list/tuple comparison and a removed implicit import re-export. Neither was a
product rejection; corrected final programs execute successfully.

[Artifacts](research/0028-standardization-spec-artifacts.json) retain the original
manifest, raw comparisons, earlier attempts and failed records. Duplicate data
has explicit reconstruction; both source archives are reproducible from pinned
Git using the [archive identities](research/0028-standardization-source-archives.json).
[Execution instructions](research/0028-standardization-reproduction.txt) use the
five programs published in `scripts/verify_standardization/` and fresh output paths.
[Recovery instructions](research/0028-standardization-artifact-recovery.txt) verify
the retained archive without generating new tasks.

Scoped review totals: Standards zero hard / one advisory; Spec zero product
findings. GPU, actual browser/Colab interaction, supported workloads, throughput
and scientific comparisons remain unverified. The
[joint-sampling design](research/0028-pipeline-sampling-integration.md) identifies
the next integration seams and proposed unrun oracles. Its shared context,
transactional bindings, source-name adapters and single ancestral fit are required
next work, alongside the full accepted issue 28 deliverables.
