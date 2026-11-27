**Exact-main bounded Spec PASS — b8eec282efb22cb3a5bbe426e3c5feb1bf70a6e3.**

Missing/partial within function/root raw capture: none found. Scope creep: none found. Implemented-wrong: none found.

Fresh execution against the exact main archive passed 280 cases: all eight functions, four roots, both covariance bases and five engines; quadratic width 23, both RFF branches, every cheap family in both product positions, all four MLP activation placements/shift rows and singleton discretization. Candidate captured/uncaptured realizations and next RNG draws match the independent original-base oracle exactly. All 1,615 original raw calls match law/parameters, shape/dtype, values, stream, engine, global sequence and pre/post state; trusted-generator replay passes.

Fresh adversarial replay passed 39 groups, including 160 last-draw-budget failures with earlier ledger records preserved, late validation, binding changes, aliases, external mutation preservation, unrecorded draws, nested identities and admission. Stream counts 0/1/64/65 behave as declared.

Source equivalence: main `src/` equals a435427; main `src/` and `tests/` equal 6a31732. Equivalence to a435427 is source-only: the repair replaced a private-helper test with public-ledger coverage and removed six uncaptured mapping-mutation cases. The independent oracle retains both modes. Original Standards rejection and repaired Standards reports are included byte-for-byte; earlier matrix REJECT history remains intact.

The contract requires “all construction, fit and calibration draws and new records are discarded; earlier committed records remain” (docs/issue-28-function-raw-capture.md:36–37). This replay supports that bounded behavior, not whole #28 acceptance; lines 53–57 retain unfinished graph/ownership/ancestor-fit/delivery work.

Full suite on main NOT_RUN here. Parent reports 1455 passed/1 skipped at a435427 and 11 repaired tests passed; these are attributed results, not exact-main full-suite evidence. Resources NOT_MEASURED; browser/Colab/GPU NOT_RUN. Main checkout unchanged, with only untracked `.venv`. Archives, scripts, outputs, environment and SHA256SUMS retained.

The portable `replay.py` was executed once in a fresh external directory. It extracted both exact archives and freshly reran original-base instrumentation, all candidate cases, comparisons and adversarial checks; all ten steps passed. Exact commands, environment, receipts and outputs are retained in `replay-results/`.
