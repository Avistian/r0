# Sampled numerical node pipelines

`GeometryPipelineRequest` opts into independently fitted numerical node pipelines
inside the existing geometry construction path. `sample_geometry_graph` returns a
`SampledPipelineGeometry` with complete named plans, calibration inputs, frozen
fits, raw provenance and four initial/final RNG snapshots. The result supports
`recipe_from_dict`, `generate_task`, `explain_generation`, `with_task` and paired
task-control changes through the existing public APIs. These task controls are
not causal do-interventions.

```python
from r0.modules import priors

request = priors.GeometryPipelineRequest(
    topology=priors.AttachmentGraphPrior(nodes=3),
    task=priors.LinearRecipe(rows=32, context_rows=16, features=2,
                             signal_features=2, active_features=2),
    calibration_rows=16,
    pipeline_prior=priors.JointPipelinePrior(
        channel_importance=True,
        l2_normalization=True,
        node_importance=True,
        gaussian_noise=True,
        weight_family="polynomial",  # alternatively uniform
        converter=priors.JointNumericalConverterPrior(
            allow_kumaraswamy=True,
            allow_activation=True,
            disallow_target_warping=True,
            variant="readout-only",
        ),
    ),
)
# Pass separate named_rng, raw_rng, calibration_rng and fit_rng to
# priors.sample_geometry_graph(request, ...). Resource limits remain explicit.
```

Defaults choose the existing broad mechanism catalog; a chosen combination may
exceed declared construction limits or encounter a genuine numerical failure.
There is no retry, fallback family or silent narrowing. `noise_scale` must remain
zero for this opt-in request: Gaussian sigma comes from its named source prior.
Authored noise laws remain available through the explicit graph API.

## Source contract and adaptations

[TabICLv2 Appendix E](https://arxiv.org/html/2602.11139v1) and the tracked
[node-pipeline sampling audit](research/0028-node-pipeline-sampling-laws.md)
provide the source laws and pinned source links. Main source:
[`_node_function.py` lines 20–73 at 9530bd65](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L20-L73).
Inventory mappings are `I2.node.02`–`.05`, `I2.converter.01`–`.03`,
`.02.graph`/`.03.graph`, and `I2.measurement.01`–`.03` in
[source inventory 06](research/0028-source-options-06.json).

For each node the builder first materializes its root/function mechanism, then
samples a **fresh independent ancestral population** through the completed parent
states and the current function. Function-fitting rows and pipeline-fitting rows
are separate populations. The pipeline fits in this order:

1. If channel importance is enabled, fit per-channel population Standardize
   (`1/(std+1e-4)`) to the actual function outputs. Sample the existing source
   polynomial or uniform positive weight vector and normalize its RMS to one.
   Disabling importance disables both this Standardize stage and its weights.
2. Optionally fit one whole-node L2 factor to the weighted outputs:
   `1/(sqrt(mean(sum(channel_values**2)))+1e-8)`. There is no division by width.
3. Fit each reserved numerical channel's identity, Kumaraswamy or standard
   activation converter to that exact post-L2 channel. Activation shift-row draws
   are retained on the fit stream. Converter channels are distinct, so their
   per-channel fits do not depend on mutations of another channel.
4. Capture readouts with the selected variant, add independent Gaussian innovation
   to propagated state, and multiply that state by node importance. Descendants
   calibrate on this final state, including ancestor innovation and importance.

The existing shared named context binds channel-weight hyperparameters to matrix
weight preferences. `gaussian_noise_sigma` uses logarithmic bounds `[1e-5,1]`;
`node_importance` uses `[0.1,10]`. Local, shared-meta and global modes retain the
existing numerical-preference semantics; meta requests are not described as
simple log-uniform draws. Weight scalarization is an explicit prior control.
The source's transform-time redraws are **not** reproduced: this version freezes
mechanism parameters once and fits independent ancestral populations, as required
by the accepted specification. RNG algorithm/order equivalence to upstream is
not claimed.

| Converter variant | Propagated channel before innovation | Captured readout |
| --- | --- | --- |
| `historical-tuple` | transformed | copied untransformed channel |
| `historical-graph` | transformed | transformed channel |
| `readout-only` | untransformed | transformed channel |

Readouts in all three cases are captured before innovation/node importance.
Historical tuple versus graph labels preserve the documented aliasing distinction;
corrected readout-only behavior is a named adaptation. Restricting feature group
exactly `y` records forced identity without a categorical draw, avoiding incompatible
shared catalogs. This is also an explicit adaptation, not source draw equivalence.
Default r0 pipeline controls enable L2 and corrected readout semantics and permit
both nonidentity converter branches; they do not claim upstream default matching.
No upstream code was copied.

## Nested archive version and admission

The outer discriminator remains `sampled-geometry-graph-v1`, so generation and
inspection transport remain unchanged. Its request explicitly selects
`geometry-pipeline-request-v1`; the additional `pipelines` array contains
`independent-node-pipeline-realization-v1` records and their versioned plans/prior.
The decoder dispatches on the nested request version, requires the matching exact
record fields, and reconstructs the entire construction before accepting equality.
A plain geometry request rejects pipeline fields; a pipeline request requires
pipeline records. Existing geometry-v1 records have no new default fields.

Each node owns a separate registered `pipeline:N` scope including event-free
weight scopes, converters and activation children. The graph retains exact final
pipelines/noise in its authored runtime binding as well as the sampled pipeline
records. Independent calibration identities/digests and ledger intervals bind each
fit to its actual ancestors. Replay compares every decision, raw draw, fit and
four-stream end state. A failed construction leaves all caller generators intact.

Pipeline raw draws use the **same** graph ledger and remaining topology/observation
budget. After frozen equations are installed, their final graph admission is
recomputed and the already charged raw ledger must also fit those final caps.
No pipeline stage gets a new raw allowance. Additional retained pipeline fits and
stage buffers reserve `8*(calibration_rows+64)*total_channels` numeric slots and
`128*reserved_slots+16384*node_count` bytes from the enclosing geometry budget
before raw construction. This covers duplicated fit matrices/constants, plans and
live stage buffers conservatively; it is not a measured process-RSS guarantee.
Existing function/root/graph/prefix admissions remain in force.

## Worked calibration row

The retained author walkthrough uses named/raw/calibration/fit seeds 100/101/102/103,
three attachment nodes, six independent fit rows, and `historical-graph` converters.
For root 0 channel 0 its first calibration row is:

| Stage | Value |
| --- | ---: |
| Frozen root/function output | -0.2149022782 |
| Independent Standardize | -0.1669973827 |
| Positive channel importance | -0.0242070993 |
| Whole-node L2 | -0.0139770777 |
| Kumaraswamy conversion | 0.9776850399 |

The whole-node L2 factor is 0.5773958068. Gaussian sigma is 0.0002731537 and node
importance is 0.6907534901. The displayed converter output is the captured readout;
innovation and node importance affect subsequent propagated state. Node 1 instead
uses a standard activation converter; node 2 uses Kumaraswamy. Raw calibration
matrices, stage arithmetic, full archive and repeated-row inspection are retained
in `docs/evidence/issue-28-sampled-pipelines/worked-*`. NumPy independently computes
Standardize and whole-node L2 here; converter evaluation uses the frozen public
runtime. This is an author arithmetic trace, not an upstream distribution study.

## Evidence and remaining scope

Public tests exercise complete archive/generation/inspection/resplit/task-control
pairs, independent Standardize/channel-RMS/L2 arithmetic, three converter variants,
readout timing, final-parent calibration counterfactuals, empty topology prefixes,
single-row fits, local/meta/global sharing, target restrictions, five engines,
tampered archives and combined-resource rollback. Source boundary failures remain
retained in ignored `docs/evidence/issue-28-sampled-pipelines/`.

The existing Cauchy archive compatibility fixture remains 45155 bytes with SHA256
`dd4200dc00ade9449a55109eb3fbc970302d667e8794b25c6c0bf60aca79fd27`; canonical bytes
and remaining raw caps match the prior candidate exactly.

This slice does not add explorer/Colab controls, native categorical variables,
non-Gaussian sampled innovations, separate-parent aggregation, fitted DT/RF/
ExtraTrees/XGBoost/GBDT mechanisms, or residual standalone networks. It does not
claim full #28, browser, scientific distribution or publication acceptance.
