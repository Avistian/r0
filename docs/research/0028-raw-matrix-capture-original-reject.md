SPEC REVIEW: REJECT (bounded public raw-recorder slice)

Candidate ccaf90637bc19a8e1173cc5fc77946a206b5e223; base 97cd0233726fed230a570435ab34dfbaf8e63556. Preflight and final status: only untracked .venv; candidate unchanged.

Incorrect — P2: raw_draw_records.py:89-91 applies the discrete positive-integer restriction for `high` to the explicitly supported continuous `uniform` law. Through public RawDrawLedger.transaction/draw, finite ordered bounds low=-0.25, high=0.75 reject with “raw discrete bounds must be positive bounded integers”; direct NumPy produces valid values. Even high=1.0 is rejected. Restrict the discrete bound validation to its respective laws while preserving finite ordered uniform bounds. Retained probe_uniform.py reproduces against the exact checkout and an independently extracted candidate source archive. Rejection correctly preserves ledger and caller RNG.

Contract relationship: docs/issue-28-raw-draw-capture.md:16-18 promises explicit distribution/parameters; raw_draw_records.py:70-86 declares uniform and its finite ordered bound rule. Accepted spec docs/specs/0028-causal-network-generators.md:81-87 requires uniform-root breadth and explicit mathematical compatibility. This finding concerns the newly exported generic recorder, not an observed regression in the five matrix materializers, which currently use other raw laws.

Missing/partial: Full graph capture, function/root/prefix integration, JSON decoding, typed owner validation and graph-wide admission remain explicitly unfinished (candidate contract lines 8, 19-21, 35-36, 48-51); these are not surprise regressions or grounds for full-issue acceptance. No scope-creep finding.

Independent verification stopped at the first concrete defect. Broad family/engine baseline comparisons, nested-failure matrix, full tampering/cap matrix and fixture comparison remain NOT_RUN by this reviewer. Author/parent checks are not reclassified as independent evidence. No full issue verdict, resource measurement or scientific acceptance.

Evidence: receipt.json pins source, interpreter hash, Python 3.12.3/NumPy 2.5.5 (raw log), command, environment and exit 1. replay.sh extracts candidate.tar and repeats the same failure; portable-replay.exit is 1. Baseline source, candidate diff and SHA256SUMS retained. No downloads, cloud calls or product edits.
