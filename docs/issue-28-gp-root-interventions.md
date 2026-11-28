# GP-root causal interventions

`intervene_gp_root_dag(source, change)` applies an explicit
`CausalInterventionRecipe` to a retained GP-root graph. Sources may be a plain
`GPRootDAGRecipe`, an independently fitted `GPRootCalibration`, or a
`GPRootCalibrationQuery` with its full prior query history. The change must name
the exact authored graph in the current source. Functional, native scalar,
native vector and converted graphs retain their respective intervention rules.

Matched changes reuse the complete joint GP draw. Population changes retain the
GP request and draw a new complete population with a deterministic identity
derived from the current identity and the requested population index. A derived
identity colliding with retained history is rejected; it is not retried. Both
modes retain the full original source and any independent target fit. They never
fit target normalization on intervened or query rows.

Partial and full hard clamps, soft functions and supported stage changes use the
existing authored intervention semantics. A bound GP root's input geometry must
stay compatible with its binding. Replacing its scalar root law is rejected:
changing the joint GP law requires an explicit population edit.

Use `generate_gp_root_pair(source, change)` for a before/after `TaskPair`.
`compare_gp_root_pair(pair.before, pair.after)` validates the full source and fixed-fit custody,
then reports the comparison. Generic generation, recipe decoding, generation
inspection, pair comparison and pair archive save/load recognize these records.
Repeated inspection rows retain their order. GP covariance receipts describe
the full population; fit statistics identify their separate fit population.

For calibrated functional tasks, the fixed normalization precedes the task's
regression-noise or classification adapter. Matched changes retain keyed adapter
draws; population changes use the existing population intervention key. Native
continuous targets use the normalized structural value directly. Native
categorical targets retain their declared domains and bypass numeric fitting.

`construct_gp_root_intervention({"source": source_dict, "change": change_dict},
seed=seed, task_id=task_id, limits=limits)` admits raw controls before decoding.
Identity must match the current source exactly. Admission reserves the complete
retained lineage and future population before covariance or random work.
Replay is CPU-only, with the original identity.

This API does not yet provide sampled population policies, nested
intervention-source editing or GP-law editing.
Those remain separate requirements of issue #28.

## Workbench

Prepare causal intervention on a plain, calibrated or continued GP-root dataset.
The existing typed controls select hard clamps, soft mechanisms, noise and
converter locations. Matched/population mode applies to the complete GP-root
comparison. Generation runs in the bounded worker; success stores a before/after
pair while keeping the original dataset active. Invalid controls and timeouts
retain the last successful dataset and comparison.

The comparison supports repeated-row before/after traces, export and reopening.
Opening its `after.zip` activates the intervened dataset: replay retains its
original identity, full GP source and fixed calibration. Its generation graph
can inspect query rows or, when calibrated, the separate fitting population.
Covariance views retain complete population matrices.

Nested interventions, sampled policies and generic task-property pairs remain
pending. The reopened intervention disables unsupported creation controls and
keeps replay and inspection available. Reopen its original source to construct
another explicit comparison.
