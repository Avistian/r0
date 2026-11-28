# Canonical population reference source admission

Child of frozen c0f206cda24aa5560aa132eb8d9ed9b6f0d60ecc. The independent
continuation review remains REJECT in
`/tmp/r0-28-population-reference-review/docs/evidence/issue-28-population-reference-complete-independent/`:
a conflicting duplicate seed inside reference source_json caused10 snapshot
restores before eventual rejection; the valid control also restored10.

The reference decoder now parses source_json with recursive duplicate-field and
nonfinite-constant rejection, then checks exact canonical JSON encoding before
source identity, geometry admission or restoration. Equal duplicates, whitespace,
alternative key ordering and escaped-key encodings reject early as well. This
moves an existing final replay requirement to admission; valid canonical records
and their generation semantics remain unchanged. Finite overflow representations
are rejected by the canonical encoder before replay.

Public decoder tests include the original genuine RED, malformed encoding
variants with a restore sentinel, and complete valid receipt equality. The
outside original probe and20-preset complete task/reference artifact comparison
are retained under ignored
`docs/evidence/issue-28-population-reference-source-json/`.

This repair does not accept the original whole reference slice. The prior
geometry repair has separate root verification. Remaining independent original
reference numerical/browser work must continue only after independent review of
this delta. Calibration-wrapper integration remains REQUIRED. No full suite,
cloud work or source evaluation changes were made.
