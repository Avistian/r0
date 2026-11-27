# Independent SPEC review: bounded ACCEPT

Reviewed HEAD: `ae815a6f44205b13ea39654ee36df5ce3099a14d` in `/tmp/r0-28-graph-admission`.
Scope: authored frozen-function causal graph slice, against `docs/issue-28-functional-graph-integration.md` and the active `docs/specs/0028-causal-network-generators.md`. Full diff: `de1bead86d51721fd84d84598b7bea2800ec8885...HEAD`; latest repair: `31a841b8097b8156eb23fc78a43263cad887fe2d...HEAD`.

No unresolved defect or scope creep found within the inspected and executed bounded cases. Both prior findings are independently repaired: constant internal mechanisms have explicit sample diagnostics; covariance validation performs scalar checks before aggregate admission, with no numeric array allocation observed on rejected channel, total-channel, or working-byte caps. Both historical REJECT bundles remain unchanged.

Independent executed evidence:

- 32 variable-input equations: eight functions, four root bases, nonlinear descendants, JSON roundtrips and repeated/out-of-order inspection.
- Heavy-tail moment bounds, shared-input cubic products, internal MLP normalization, zero/factor/source paths, three converter variants and literal noise ordering.
- Malformed nested containers, unknown fields, raw traversal/storage limits, caller/payload caps; transparent NumPy/Torch allocation observers; numeric metadata counts, ancestry budget and exact working-byte gate checks.
- 36 pinned binary64 covariance-validation cases, including uniform/polynomial weights, extremes, underflow, rejection parity and one-ULP corruption. Seven pinned base draw/emission combinations and eight standalone root materialization/emission/reconstruction paths.
- Twelve session save/open/replay cases across four bases and three task types, with literal equations and topological trace checks.
- Four pinned legacy affine arithmetic cases and six generation/RNG comparisons across task types and noise positions.
- Internal MLP saturation, inactive pipeline and converter domain-clamp diagnostics.

These are independent programs, not author-test results. One initial verifier fixture passed a NumPy scalar where a built-in float was required; its failed program/logs are preserved, and the corrected program passed. No product defect was hidden by that correction.

Resource figures remain estimates, not measured RSS or OS caps. No broad-suite rerun, GPU, throughput, local-browser or hosted-Colab evidence is claimed. Sampled graph ownership/calibration, target normalization and remaining full-issue delivery remain unfinished. This verdict does not close issue #28.

`replay.py` accepts explicit source root, output directory and expected HEAD. `portable-run/` records its exact-head execution using the public decoder export; internally instrumented allocation and pinned-comparison checks are separately visible in their programs. `SHA256SUMS` seals retained scripts, commands, logs, payloads, archives and source pins.
