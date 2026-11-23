# r0 implementation reconciliation

Issue [#20](https://github.com/Avistian/r0/issues/20). The next implementation is #21, the single-table episode and replay contract.

## Baseline

Audited GitHub main [807da066ef3d0df1090e1ef99b7340b11eca4dc4](https://github.com/Avistian/r0/tree/807da066ef3d0df1090e1ef99b7340b11eca4dc4) in a separate detached checkout. The authoring checkout began at 51a3bebb5530e93a0926267682d9813d6a911193. Intervening changes affect CLI hello behavior and tests. Findings use the newer pinned source.

The [crosswalk](reconciliation/crosswalk.md) maps individual promises. The [inventory](reconciliation/sources.md) identifies supplied research and original ticket excerpts. The [curie review](reconciliation/curie.md) identifies transfers. The [reader walkthrough](reconciliation/review.md) records acceptance.

## Observed behavior

Invoke the public Python CLI entry point using an existing r0 environment, with PYTHONPATH pointing to the pinned checkout's src directory.

    PYTHONPATH=/path/to/pinned-r0/src /path/to/environment/bin/python -c 'from r0.cli import main; main(["--version"])'
    PYTHONPATH=/path/to/pinned-r0/src /path/to/environment/bin/python -c 'from r0.cli import main; main(["--help"])'

Observed version was 0.1.0, exit 0. Help advertised --quiet, -h/--help, --version and --hello, exit 0. Metadata came from the existing environment and agrees with pinned pyproject.toml. Fresh installation was not performed.

Source is [CLI](https://github.com/Avistian/r0/blob/807da066ef3d0df1090e1ef99b7340b11eca4dc4/src/r0/modules/cli/__init__.py). Nine ML entrypoints and lib initializers are empty. The tracked tree has no evaluation suite or run/smoke configuration. Makefile calls future subcommands that have no implementation.

## Governing decisions and enforcement evidence

- AGENTS requires synthetic-only training and prohibits TabPFN teacher signals. pyproject.toml declares import-linter boundaries. Configuration establishes the declared boundary; scientific behavior of the empty modules remains unimplemented.
- CODING_STANDARDS requires explicit random generators, context-only normalization and query-only scoring. T1's global seed helper prescription is superseded by explicit replayable generators in #21 and recovery in #32.
- AGENTS and ADR 0005 require strict prediction-time visibility. The featurizer is empty. #40 implements visibility and counterfactual evidence. #39 defines future label windows separately from prediction inputs.
- AGENTS protects protocol/evaluation paths. ADR 0003 remains draft and the suite is absent. #25 and #41 own human-reviewed protocol freezes.
- ADR 0004 documents guarded compute. agent-train.yml pins ml-factory 462fbd57fca0e3a1b9e18c4eeb453bebe0f1bc2e and declares concurrency. Provider caps, secrets, kill-switch state and hosted execution were not inspected or exercised. #34 owns operational evidence.
- ADR 0009 chooses committed result records and release checkpoints. Artifact implementation belongs to #34, #38 and #42.
- Requirements ask to study curie; AGENTS forbids copying code. Independent implementation informed by concepts and behavior satisfies both. A software license does not override that repository instruction.
- ADRs 0001, 0002, 0003, 0006, 0007, 0008, 0011 and 0012 contain draft text. Explicit AGENTS constraints still apply. Draft titles do not settle architecture, mixture, baseline version or asset rights.

Sources are in the pinned repository tree linked above. No identified conflict blocks #21's strict configuration, disjoint episode and explicit-generator contract. Scientific decisions have downstream owners in the crosswalk.

## Vocabulary

GLOSSARY.md links to CONTEXT.md. Domain definitions now cover single-table and relational tasks. Historical prescriptions for DFS-lite, 40 features, NaN only for absent history, 64/256/512 shots, a frozen RelBench suite and a sub-minute smoke are preserved in the crosswalk. No new ADR is needed for this audit of agreed planning decisions.

## Completion

The fresh reader reproduces a CLI observation and traces T1, T8, T16 and one curie finding to evidence and owners. Scientific decisions may remain open downstream. Any ambiguity affecting #21 must be resolved here. Validation and the final remote comparison are recorded in the reader walkthrough.
