# Calibrated population inspection identity admission

This repair is a child of frozen `87bcd2afb4505a2890f565667f75a7b1b1959685`.
The independent review of that source remains **REJECT**: inspecting a saved
calibrated task with a changed outer seed restored 32 streams before rejecting.
The valid control restored 74 streams. Historical evidence remains in
`/tmp/r0-28-calibration-core-review/docs/evidence/issue-28-calibration-core-independent/`.

Inspection now calls the shared public cheap generation-identity validator before
source decoding. A separate inspection helper also binds the implementation name,
CPU execution, source-task fingerprint and split-dependent episode fingerprint.
The existing strict record parser rejects malformed execution headers, numeric
identity aliases and extra count/sampling/construction fields before this guard.
Historical Python/library descriptions remain accepted as provenance. Complete
recipe admission and exact tensor/task replay validation still run afterward.

Verification uses public `explain_generation`, including repeated original rows
and IDs above 2**55. Six cases genuinely failed before the change by reaching a
stream-restore sentinel. Seventeen malformed outer-record cases and a valid
historical-environment control now pass. The original outside-test probe reports
zero restores for the changed seed and the unchanged 74-restore valid control.
Retained four-family regression and functional binary archives are compared with
new replay across complete task metadata, recipe/mechanism JSON and all tensor
bytes. Raw evidence and command logs are ignored under
`docs/evidence/issue-28-calibration-inspection-identity/`.

This is an inspection admission repair, not acceptance of the whole calibration
core or issue #28. Pair/query inspection extensions must apply the same guard to
their own retained identities and source-linkage rules. No full suite, browser,
cloud work or scientific distribution experiment was run for this repair.
