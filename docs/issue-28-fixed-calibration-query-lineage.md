# Fresh query populations with retained independent calibration

`realize_calibrated_population_query(source, seed=..., task_id=..., engine=..., construction_seed=...)`
constructs a versioned `CalibratedPopulationQuery`. Its `source` retains the complete
previous calibrated graph or query envelope; its `original` contains only the new
uncalibrated row-owned query population. `calibration` and `calibration_population`
resolve to the original independent fit. Repeated Next populations retain the full
lineage rather than replacing or relabeling a frozen archive.

The request, including row count, topology, structural/root/domain/pipeline laws,
observations, target schema and limits, must remain exactly unchanged. Query
construction uses a separate protocol key over that request, explicit seed/task
identity, selected NumPy engine and recorded construction seed (defaulting to
the episode seed). The construction seed affects row-forest construction only;
root/noise/label streams still use the episode identity. It does not use realized query values or fit
statistics to seed its stream. All five existing engines are supported; the default
retains the source query's engine. Every new identity must differ from all query
ancestors and the calibration population. Replay uses the retained identity;
reusing an ancestor through the fresh-construction operation rejects.

Only the new query population is constructed. Source validation replays retained
custody and verifies existing calibration statistics; it does not fit replacement
constants or create a new calibration population. Public call instrumentation
checks that fresh construction invokes exactly one query realization and no
calibration constructor. Literal forest traversal independently verifies target
values against the retained independent population's mean/scale, including a
second query generation. Constant-channel policy and functional target adapter
semantics remain those of the original calibration.

Continuous functional, native scalar, heterogeneous-vector and converted graphs
use their existing typed emission contracts. Functional binary/multiclass adapters
receive fixed calibrated score channels. Paired interventions may use a query
baseline: they retain its complete lineage, fixed fit and original row ownership;
matched/fresh counterfactual rules remain explicit. Direct categorical codes are
not numerically normalized by this feature.

Complete lineage metadata, every query request/receipt and the independent fit
are admitted before construction or restoration. Decoder replay checks exact
same-law requests, identity uniqueness, selected engine, canonical initial stream,
raw snapshots, original inputs and full archive bytes. The lineage is bounded to
at most 64 links plus the existing metadata and caller resource ceilings; these
are implementation limits, not paper distribution laws. There are no retries,
hidden row reductions or automatic history truncation. Task/run/inspection guards
use the current query identity before replay. Save/reopen and source fingerprints
retain exact large integer identifiers.

Changing geometry or laws requires an explicit new independently calibrated
construction; it cannot use this same-fit operation. A linked recalibration
workflow remains a separate required extension. Guarded workbench Next/Replay,
calibration controls and reference composition are the next required integration
slice. A reference grid must retain the full wrapper record, evaluate its current
uncalibrated population at original row ownership, and apply the fixed transform
before the target adapter without fitting on grid/query rows. This library slice
does not claim UI, hosted, scientific or maximal-resource acceptance.
