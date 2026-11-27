# Numerical pipelines in the shared workbench

Three construction presets expose the numerical converter contracts:
`Numerical pipeline: historical-tuple`, `Numerical pipeline: historical-graph`
and `Numerical pipeline: readout-only`. Their `pipeline_prior` JSON exposes
channel importance, L2 normalization, Gaussian innovation, node importance,
weight family and scalarization, plus converter activation/Kumaraswamy switches,
target-warp restriction and variant. The same optional object is accepted by
Cauchy, attachment and motif geometry construction. Omitting it preserves the
existing plain geometry request and its defaults. Authored native variables keep
their separate construction path.

Historical tuple changes descendant state and captures the raw readout;
historical graph changes both; readout-only changes the readout and preserves
state. All readouts precede innovation and node importance. Readout-only and the
frozen independent fitting policy are named adaptations. These numerical
converters do not implement native categorical variables. See the
[sampling contract](issue-28-sampled-numerical-pipelines.md) for source attribution,
fit ordering and retained provenance. Sampled pipeline noise requires authored
`noise_scale` to stay zero; the Gaussian prior switch controls its sampled sigma.

The mechanism editor exposes each frozen numerical pipeline as JSON. Editing
constants produces a linked authored mechanism and does not recalibrate its fits
or claim a causal intervention. To refit, construct a new sampled graph. Existing
save, reopen, generation-graph inspection and dataset/edit exports retain the
sampled recipe, construction transcript and authored successor independently.

Author evidence is retained in ignored
`docs/evidence/issue-28-pipeline-workbench/`, with exact commands in `commands.sh`.
The public test first failed because the presets were absent. Subsequent tests
exercise all three variants, Gaussian noise, graph row order, pipeline editing,
saved replay and exports. The mocked Colab callback matrix adds readout-only
construction and edit/reopen to its existing native and topology cases. The real
installed Chromium helper now accepts pipeline presets and verifies retained
controls plus the pipeline editor, saved edit/export, reconstruction and mobile
layout. Existing helper arguments remain valid.

Hosted Colab: **NOT_RUN**. Combined full suite and independent acceptance remain
with the parent integration reviewer. This component does not complete issue #28.
