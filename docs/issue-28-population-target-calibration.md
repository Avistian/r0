# Independent population target calibration

`calibrate_population_graph(original, calibration_rows=..., calibration_seed=...,
calibration_task_id=..., limits=...)` returns a versioned `CalibratedPopulationGraph`.
It retains the original query population, a separately keyed same-law graph
population, and fixed per-channel target centers/scales. The calibration identity
must differ from the original. Its forest stream is canonically keyed by the full
calibration request, seed and task identifier; it cannot be relabeled as an
independent draw after reusing another construction. The original graph's frozen
mechanisms, root laws, typed emissions and node population plans remain unchanged.
Only calibration row count and its bookkeeping context size differ. No query
inputs, labels or query moments fit these constants.

The current row-owned forest contract bounds calibration to 32..4096 rows, subject
to explicit aggregate parameter/raw/workspace budgets. This is an implementation
resource bound, not a paper law. Requests outside it reject without clipping or
retry. Larger-population support remains required breadth with separate admission
work; the general target-calibration record's larger bound is not a promise that
this forest lifecycle supports those populations today.

For a functional graph, fixed calibrated score channels feed the existing
regression/binary/multiclass adapter. Observation noise or classification uniforms
use the original keyed streams. Constant channels retain identity scaling
(center 0, scale 1), with their indices explicitly recorded; no artificial unit
variance is claimed. Native scalar, heterogeneous vector and converted continuous
targets scale their already emitted continuous values directly. Categorical target
labels are never treated as numerical regression scores: requesting this numeric
calibration for direct categorical outputs rejects and leaves the original
category task intact. Calibrating categorical emission logits/probabilities is a
separate required extension, not implemented by scaling class codes.

The complete archive retains both populations, all row-owned forests/inputs/raw
records/snapshots, calibration digest and prefix identity. Decoder admission checks
both retained populations before any restore, applies caller limits, and then
replays source construction. Generated task semantics and complete original-row
traces are bound to exact replay. Trace IDs for the original population have an
`original-` prefix; a separate shared node summarizes the independent calibration
score matrix and its identity. This matrix does not use selected query row IDs.
The final signal node shows the fixed transform, followed by actual adapter
noise/probabilities/uniforms or direct continuous target selection. Complete
independent internal graph traces remain accessible by inspecting the retained
`calibration_population` itself. Task JSON and task ZIP save/reopen preserve the
full envelope and exact large integer identities. The generation runner verifies
complete calibrated task semantics before persisting custom producer output;
malformed headers or identity types produce failure receipts instead.

Core construction/replay/trace is the first slice. Matched and fresh intervention
comparisons must retain baseline calibration without refitting; explicit recalibration
needs a new linked construction. Those paired lifecycle APIs and explorer
construction/edit/reopen/reference controls remain REQUIRED follow-ons. This
slice does not advertise them as available. Hosted execution, scientific
reproduction and maximum-size resource measurements are not established here.
