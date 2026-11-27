# Causal intervention explorer

Generate or open a supported numerical or native graph, then open **Causal
interventions** and prepare controls. This creates a separate saved comparison;
the original dataset and its complete construction/fit archive remain active.
The existing mechanism editor remains a distinct authored-model workflow.

Choose a node and add hard or soft changes. Hard numerical changes accept ordered
channel indices and matching values. Native discrete nodes use their declared
category vocabulary. Soft changes accept frozen function JSON and optional
`noise_scale`, `noise_response`, and `exogenous` replacement fields. The complete
JSON array supports multiple versioned changes through the public recipe decoder;
it is also the escape hatch for supported changes beyond the convenience controls.
Unsupported versions fail without changing the active dataset or comparison.

For numerical pipelines, **state** changes the value passed to descendants while
preserving earlier captured measurements. **Readout** changes the measurement
without changing descendant state. **State and readout** explicitly changes both.
For uncaptured channels, state and readout represent the same variable. Fully
replaced mechanisms can be bypassed; partial coupled mechanisms are evaluated
before clamping. Native category clamps replace emitted values and one-hot parent
state; displayed forced probabilities are indicators, not the original model's
probabilities. Calibration stays frozen; intervention queries never fit it.

**Matched** comparisons retain original row identities and unchanged exogenous
draws, with exceptions and bypasses recorded in the card. **Population** uses a
fresh keyed draw population; matching row positions are not matching units.
Seed, task and population identities travel as decimal strings in browser controls
so values above JavaScript's exact integer range remain intact.

The shared-axis plots compare the same original row positions. Enter row indices
(including repeated or reordered indices), inspect actual before/after stages,
and select a stage to see its values and original row identities. The saved card
records changed components and draw checks. Download includes both group ZIPs,
completion records and pair manifest. Extract it and call
`r0.modules.generation.load_causal_pair(Path(directory))` to verify/replay it.
Reopening within the explorer accepts directories under its configured output root.

Creation, reopen, trace replay and export run through the guarded worker with
CPU-only execution, timeouts, input/aggregate output limits and retained attempt
receipts. A failure preserves the original dataset and previous valid comparison.
Pair files are checked together, rather than treating the per-group persistence
limit as a pair-total limit. Failed attempts may leave diagnostic artifacts but do
not publish a successful session result.

The local HTTP UI and embedded Colab adapter share these controls and session
operations. Local callback tests cover create/reopen; hosted Colab is **NOT_RUN**.
This slice exposes hard and frozen-function/noise soft changes on the base API.
[Joint vector intervention controls](issue-28-vector-interventions.md) add explicit
logical-variable selection and root/domain/pipeline-stage changes. No complete
issue #28 acceptance is implied.

Verification uses public session round trips, timeout/size/stale-identity failures,
native categories, actual Chromium numerical/native/fitted workflows, pipeline
state/readout differences, original archive preservation and browser downloads.
Raw logs, screenshots and downloaded artifacts live in ignored
`docs/evidence/issue-28-intervention-explorer/`.
