# Fixed target calibration across population interventions

`intervene_calibrated_population(base, change)` produces a distinct
`CalibratedPopulationIntervention`; `generate_calibrated_population_pair` returns
the baseline and counterfactual tasks with a replay-checked comparison card.
The `base` is the complete `CalibratedPopulationGraph`; `original` retains the
ordinary `RecursivePopulationGraphIntervention`, including the exact original
population, every changed forest receipt, node input rows and bypass ownership.
The wrapper's `calibration` property is the baseline's retained calibration.

Matched execution preserves the original row forests and keyed innovations.
Population execution uses the existing explicit population identity and stream
mapping for fresh forests/innovations, while preserving row and split alignment.
Both apply the same independently fitted baseline center/scale to the changed raw
target channels. Functional regression/classification adapters run afterward with
the correct matched or fresh noise/label streams. Continuous native/vector and
converted targets stay direct continuous emissions. Categorical mechanisms inside
those DAGs keep their vocabulary and typed state semantics. This is not numeric
normalization of categorical target codes, nor a promise of unit variance after
an intervention.

Hard, soft, root/noise/pipeline/domain changes keep the existing four-family
intervention contracts, including complete joint bypass and per-row mechanism
replacement. Calibration never causes an implicit new fit. Explicit recalibration
must be a new independently keyed construction with its own identity; a linked
recalibration workflow remains required separately.

Before any nested replay the decoder validates the exact baseline/change coupling,
both calibration populations and the counterfactual envelope, all raw/snapshot
custody, caller limits and aggregate retained metadata/workspace. Three retained
populations plus changed runtime can require larger explicit budgets than a single
calibrated task; a limit failure never filters options, retries or shrinks rows.
Pair JSON/ZIP save/reopen uses the shared causal comparison contract with complete
task semantic validation and exact primitive identities on both sides. Traces keep
the changed raw execution separate from the shared baseline calibration population
and show the actual adapter arithmetic on original selected rows.

Workbench calibration construction, pair controls and reference-curve integration
remain the next required slice. The reference implementation must preserve the
full wrapper source record, evaluate/clamp the uncalibrated runtime using the
retained original row ownership, and apply fixed calibration before the adapter.
It must never fit on response grids or transform a raw intervention clamp into a
new source-mechanism parameter. No hosted, scientific-reproduction or maximal
resource claim is made by this library slice.
