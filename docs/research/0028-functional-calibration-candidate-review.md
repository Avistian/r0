# Independent SPEC review: ACCEPT, bounded target-calibration slice

Candidate `ca50743d1ebc2479d4c7a57fb4434141910270eb`; base `ae815a6f44205b13ea39654ee36df5ce3099a14d`. Reviewed `git diff ae815a6f44205b13ea39654ee36df5ce3099a14d...HEAD`; sole commit `ca50743 feat(priors): calibrate functional graph targets independently (#28)`. Preflight and final status: only `?? .venv`. Candidate left unchanged.

**Implemented incorrectly:** no defect found in the bounded independently executed checks. Decision 12 (spec lines 76–80: “independent reproducible calibration samples, retaining the resulting constants”) is supported by a separate scalar NumPy/hash-stream oracle for nonlinear downstream values, signal-dependent structural noise and pre-noise converter capture. Emitted row count, split/context and observation-label noise changes preserved the entire calibration record. Inspection used retained constants without fitting. Shared session create/save/open/replay passed.

Decision 13 (lines 81–87: “required finite moments” and “Preserve valid extreme draws”) is supported by strict Student-t second-moment and squared-output fourth-moment boundary checks, positive/negative multiclass channel calibration, `1e300`/`1e-300` normalization cancellation, and a literal ±`1.7e308` tree whose opposite-sign subtraction overflows but normalized values remain finite. Ten malformed-card cases and exact declared admission threshold rejected appropriately.

Version-one compatibility passed byte-for-byte against the base source for 12 cases: four roots × regression/binary/multiclass; 129 retained files cover tensors, recipe/mechanism JSON, identities and inspection traces.

**Missing/partial:** whole-issue requirements remain incomplete as explicitly declared: sampled graph construction and ancestral fitting; broad family/workbench/intervention delivery (lines 28–64); measured resources/CPU-GPU/browser/Colab evidence (lines 122–130); complete option-level implementation (lines 142–167). These are author/verifier execution work, not new scope decisions. Admission checks establish declared bounds, not measured RSS. This review does not accept all of #28.

**Scope creep:** none identified. The ownership research is labeled a proposal, not implemented coverage.

Evidence: `probe.py`, `supplement.py`, `v1_snapshot.py`, stdout/stderr, literal inputs/oracle populations, source archives, environment pins and `SHA256SUMS`. An initial verifier harness mistake opened archive ID 0 instead of 23; original program/error retained as `probe-attempt1.*`, then only the harness was corrected. No product failure was overwritten. Full suite, browser, Colab, GPU and scale checks: NOT_RUN by this verifier.
