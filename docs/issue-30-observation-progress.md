# Issue 30 generator implementation progress

The full issue remains open. These are component-level implementation records,
not acceptance of the complete observation or mixture portfolio. UI/explorer
presentation belongs to #126; generator algorithms and headless evidence stay here.

| Requirement | Current component and remaining work |
| --- | --- |
| O30-01 Typed observations | `TrainingFeature` now transports continuous/count/binary/categorical/ordinal columns, including singleton categories and declared ordinal order. Type-producing transformations, encoding maps, raw-value provenance and sampled/shared preferences remain open. |
| O30-02 Marginals/measurements | An ordered measurement kernel now implements affine/power/log/bounded/periodic/exponential maps, clipping/rounding, Gaussian/shared/heteroscedastic/Student noise and additive outliers. Task/CLI integration, fitted transforms, typed encodings, redundancy/projections/interactions and further families remain open. |
| O30-03 Missingness | `TrainingTask.missing_mask` and archive transport now preserve explicit missing cells for all feature kinds. A bounded MCAR/MAR/self-target-latent MNAR/row/block kernel now retains probabilities, value binding and RNG snapshots. Single-stage CPU observation tasks now retain privileged raw inputs/drivers and verify mask replay on archive loading. CLI/guarded requests now apply masks after source generation and validate observation receipts on resume. Additional structured families, automatic latent-driver extraction, stronger latent provenance and indicator policies remain open. |
| O30-04 Targets/difficulty | Existing task adapters are reusable; the issue-wide observation/label-noise and difficulty inventory remains open. |
| O30-05 Shapes/regimes | Existing bounded task sizes are reusable; observation-regime hyperpriors, shared preferences and rare-regime sampling remain open. |
| O30-06 Composition | Ordered observation pipeline, compatibility/exclusion rules and joint coverage remain open. |
| M30-01 Outer mixtures | Production mixture-policy inventory and denominator accounting remain open. Per-table tree ensembles in #29 do not implement this. |
| M30-02 Curricula/adaptation | Policy state, approved feedback signals, probability floors and adaptive/curriculum candidates remain open. |
| M30-03 Rejection/accounting | End-to-end proposal/transformation/rejection/acceptance/queue/consumption accounting remains open; external consumption depends on #33. |
| M30-04 Diversity | Joint coverage, duplicate/opportunity-cost reports and downstream comparison remain open; downstream benefit measurements depend on #36. |

## Typed and missing-cell transport

Count columns store nonnegative integer-valued numbers, decoded as integers.
Ordinal columns store vocabulary indices and preserve their declared order in
`Episode.features`. Categorical/ordinal vocabularies may contain a single category;
constant columns are not repaired. Binary columns keep their two-label encoding.

`TrainingTask.missing_mask` is an optional boolean tensor with exactly the input
shape and device. When supplied, it must identify exactly the input NaNs, and all
unmasked cells must be finite. A finite sentinel cannot masquerade as a missing
cell. Masked category/count values are excluded from code/value validation; all
present values still obey their declared types. Episode conversion maps missing
cells to `None` and never passes NaN through the immutable episode contract.

Archives store context/query mask arrays alongside the corresponding predictor
inputs. Loading requires both arrays with matching boolean shapes and validates
them against reassembled inputs. Masks describe the observed missingness pattern;
they do not contain latent mask-generation drivers. With no mask, the existing
archive field layout is unchanged. Existing schema records retain the same two
fields (`kind`, `vocabulary`); older archives do not need a new schema flag.

Focused checks cover count/ordinal/singleton-category values, exact missing masks,
archive reopening, episode values/order and invalid counts/codes. Native-variable
and existing archive regressions are also run. This is transport support, not a
claim that a trained model handles missingness or that an observational generator
has yet been delivered. Transformation identity, raw-value custody and replay must
be integrated before describing altered inputs as a reproducible observation task.

[Initial missingness laws and integration limits](issue-30-missingness.md).

[Measurement laws and integration limits](issue-30-measurements.md).
