# Native typed variables in the explorer

The local page and shared Colab page expose **Native mixed graph (authored)** in
construction controls. It starts with continuous, binary and categorical variables
and a direct categorical target. Equations, domains, vocabularies, emissions,
observed channels and the single target channel are explicit JSON controls.
This is an authored native graph, not a sampled native structural prior.
Construction seed does not sample its equations; dataset seed controls emitted rows.

The explorer preserves typed feature schemas, declared vocabulary order and
`native-variable-v1` target origin in inspection and exported identity metadata.
Tables and CSVs decode categorical values. Binary feature values remain 0/1;
their declared vocabulary is retained. NPZ arrays retain their numeric vocabulary
indices. Predictor CSV query targets stay empty; diagnostic truth is separate.
Category distribution charts show declared categories rather than continuous bins.
Softmax emission probabilities and deterministic argmax/threshold indicators have
separate labels. Native targets bypass the numeric target adapter.

The mechanism editor can author a linked native graph with changed equations,
domains or readouts, then save, reopen and export the comparison. This does not
claim causal intervention or exogenous-draw alignment after structural changes.
Changed-target fraction is unavailable when target vocabularies differ. Native
adapter pairs and response curves are unavailable. Graph explanation retains
requested original-row order, including duplicates.

## Verification and scope

Author evidence lives in ignored `docs/evidence/issue-28-native-explorer/`.
`commands.sh` retains the commands. The first public test failed before native
construction was exposed; initial serialization-assertion, command-path and direct-target fixture failures
are preserved alongside subsequent checks. Public session tests cover mixed,
continuous-target and binary-target construction; typed exports; explanation;
linked native domain edit and replay. Mocked Colab callback tests use the same
session actions. Real installed Chromium walkthroughs retain screenshots, ZIP
exports and receipts for construction, categorical charts, native edit, reopen,
export and a mobile viewport. `numeric_compatibility.py` and before/after logs
compare source/episode identities, inspection JSON and all replay ZIP members for
three legacy numeric task types. ZIP container timestamps are excluded.

Hosted Colab: **NOT_RUN**. Full-suite integration and independent acceptance belong
to the parent integration review, not this author check. This component does not
complete issue #28; sampled native geometry, broader typed transformations and
scientific acceptance remain outside this delivery. The prerequisite native
library's known private test imports are unchanged here and repaired separately.
