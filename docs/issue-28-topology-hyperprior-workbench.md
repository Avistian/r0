# Topology hyperprior workbench

The explorer can now construct authored-template and joint-function topology
hyperprior graphs through the bounded generation worker. The complete public
`HyperpriorGraphRequest` or `JointHyperpriorGraphRequest` remains in the saved
recipe; the workbench does not translate it into a narrower surrogate prior.

Choose a **Topology hyperprior** construction, load the topology controls, edit
and explicitly apply the draft, and adopt its displayed causal caps if desired.
Construction is transactional: failed admission, numerical errors or timeouts
leave the active dataset unchanged. The separate timeout and output-byte controls
remain in force. Conservative working-memory estimates are not measured RSS.

## Exposed choices

- Fixed node count or uniform log node count, floor/ceil/nearest-even rounding,
  Gamma shape/rate, component count and both named Gamma mappings.
- Five NumPy engines. A joint construction seed creates five independent child
  streams for topology, named decisions, raw draws, calibration and fitting.
  Authored construction uses one topology stream. Original/final snapshots and
  raw records remain in the exact archive; replay restores them, while Next
  retains the realized construction and advances task identity.
- Eleven joint catalog presets: lin, quad, gp, tree, disc, em, mlp, prod,
  kumo-prod, default and default,kumo-prod. The source token control additionally
  permits tabpfnv2, named reducers and explicit comma-separated catalogs with
  duplicate slots. The complete request remains editable.
- All five matrix branches, four root families, vector and scalarization laws,
  frequencies, discretization controls, root function priors, internal noise,
  channel width, calibration rows, meta/local/global modes, shared/refresh
  categorical policies and both global numerical policies. Complete nested
  records are exposed as JSON. Conflicting shared-name contracts fail explicitly.
- Authored root/edge templates, task adapters, pipelines, causal and raw budgets
  remain editable in the full request JSON. Sampled template receipts remain
  attached to their original functions rather than being converted to authored
  approximations.

The log-size rounding and Gamma-to-probability functions are the named r0
reconstructions described in `issue-28-topology-hyperpriors.md`; this interface
adds no source-equivalence claim. Broad catalog presets keep their full support
and display larger conservative caps. Selecting them does not automatically raise
caller caps, filter options, retry numerical failures or clip source draws.

## Validation and custody

The new public authored request decoder builds a discarded metadata preview,
checks full support and original receipt costs, validates exact outer fields and
nested snapshot/raw-record shapes, and checks sampled convolution cardinalities
and selected geometry before restoring templates. The existing strict joint
request decoder is exported and reused. Both worker paths finish request
admission before constructing their RNG streams.

The existing save/open, exact replay, ZIP export, original-row graph inspection
and repeated-row controls work with these archives. Topology graph inspection
runs in the guarded worker. The retained-record panel displays the original JSON
text unchanged, preserving large integer identities and snapshot payloads.

## Evidence and boundaries

Public RED/GREEN covers missing joint/authored construction and full catalogs,
plus a sampled-template final snapshot header that previously reached restore.
Focused tests cover valid sampled-template custody and malformed header,
cardinality, selected address and raw shape rejection before restore. The ignored
`docs/evidence/issue-28-hyperprior-workbench/` packet retains raw checks, browser
receipts and numerical scripts. No full suite ran in this worker.

Outside-test arithmetic checks use scalar DAG equations and retained root inputs,
cover five engines and four root laws, full JSON/ZIP custody and five-stream
rollback. They are not distribution-equivalence tests. Browser evidence covers
actual constructions, named mappings, replay, Next, repeated rows, rollback and
a downloaded ZIP independently checked against literal authored arithmetic.
Full finite catalog admission is checked; every possible sampled branch has not
been realized in the browser.

Native/converted/parent geometry combinations, sampled intervention combinations,
independent population target calibration and remaining issue-28 breadth remain
required follow-ons. This work does not establish complete issue-28 acceptance or
change issue-30 selection and issue-39 row-axis boundaries.
