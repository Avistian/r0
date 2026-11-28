# Fixed-calibration original-row references

`evaluate_population_graph_reference` and the raw
`population_reference_from_source` entrypoint accept retained calibrated graphs,
fresh-query lineage and fixed-calibration intervention envelopes. They return the
distinct `CalibratedPopulationReference` record, version
`calibrated-population-graph-reference-v1`. The original uncalibrated reference
version and its serialized bytes remain unchanged.

Each point reuses the selected original row's retained forest at every replaced
node and executes with the complete original row shape. Observed variables are
clamped under the existing numeric/native/capture semantics; all other observed
values stay anchored. Matched and fresh comparison sources retain their own
recorded populations and stream mappings. No additional population is constructed
for a grid, and no calibration is fitted to its points.

The result records raw structural target scores, the retained calibration
fingerprint, calibrated scores, actual adapter/domain probabilities and realized
targets. Fixed scaling is applied before the functional task adapter. Continuous
native and converted readouts preserve their typed semantics. Classification links
and original-row noise/label draws retain the same stream mapping as the saved
source. These are causal original-row references, not conditional means or a
prediction that empirical query variance is exactly one.

`source_json` retains the complete calibrated wrapper, including current query,
full ancestry and independent fit population. Reopening validates the version,
full source and fingerprint, exact identities, mode, category/score geometry,
raw/calibrated arrays and derived values. Metadata admission reserves the entire
source and fit history, original-shape runtime and both grid score matrices before
any nested construction replay. Existing row-work/grid/caller limits still apply;
no truncation, resampling, retry or automatic budget increase is used.

The workbench selects active, comparison-before or comparison-after populations.
It can display raw structural scores, calibrated scores, class probabilities and
realized targets separately. Curve JSON retains the canonical `reference_json`
for public library reopening and the full original source. Timeouts preserve the
active dataset and comparison. Converter capture versus representation controls
keep their existing meanings.

The original reference lineage includes separately reviewed repairs for strict
source JSON, target geometry and worker metadata. Those bounded checks do not
imply independent acceptance of every underlying reference combination. This
calibrated extension requires its own independent review. Hosted/GPU, scientific
source reproduction and maximum workloads are not claimed. Linked recalibration,
task-only adapter comparisons and other explicitly pending issue-28 breadth remain
required beyond this reference composition.
