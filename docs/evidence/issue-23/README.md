# Generator diagnostics walkthrough

This is construction/accounting evidence for #23, not evidence that r0 outperforms a peer generator. Production generator diversity remains work for #27–30. The [audit](../../research/0023-prior-audit.md) records source support and unknowns; the [implementation crosswalk](../../research/0023-implementation-crosswalk.md) distinguishes executable subsets from the broader catalog.

Run from the checkout with its environment:

```sh
uv run r0 diagnose-priors --config configs/diagnostics/coverage.toml --out /tmp/r0-prior-run
uv run python scripts/prior_walkthrough.py --out /tmp/r0-prior-walkthrough
```

Use a fresh output path. Each run retains configuration, expectation manifest, per-proposal raw observations, mechanisms, stream identities, predictor views, report, coverage table and file checksums. Timings are observed local CPU durations, not portable performance criteria. Semantic fingerprints exclude timing. No training, learner fitting or W&B call occurs.

## What to inspect

The default has 72 proposals, two per family/noise/task-type cell. All 72 should generate, validate and be retained. The biased selection uses exactly those proposals, keeps all 72 valid, and retains 36. Its 18 added-noise cells report selection loss; those tasks are not invalid. Inspect `reference/coverage.md`, `biased/coverage.md` and their JSON reports for denominators and reasons. Inspect each record's relevant-feature count to follow sparse/high-noise examples through selection.

`worked-calculations.json` contains 12 independent scalar row calculations: noisy regression and noisy multiclass for each family. For GP it reconstructs a Cholesky factor from the declared covariance rather than calling the generation helper. Neural cards retain each intermediate layer; SCM cards retain ancestors; tree cards show each branch; interaction cards state the rule. Each card names its full raw proposal file and shows expected/observed targets and differences.

Retained results and archive checksums are in [observations](observations.md); the [review receipt](review.md) preserves initial rejections and independent delta confirmations.

The source-backed path is audit C06 → PFN 2112.10510v1 RBF component → `diagnostic-v1` GP profile → GP proposal card → family/noise/type coverage. The finite eight-coordinate domain is an explicit restriction; this is not a checkpoint recipe reproduction. The proposed r0 path is C08/C20 → bounded gated interaction → interaction cards and coverage. The implemented gate is `product(x[1:degree]) if x0 > 0 else -1`; the richer signed linear-plus-gated example remains a future hypothesis, not an implemented formula.

The walkthrough separately records exact content under a new partition, fixed mechanisms with new observations, row/class equivalences, altered query labels, changed target with intact generating rule/noise, missing proposals, and retained generation failures. A control's diagnostic `FAIL` is expected evidence only when the surrounding walkthrough explicitly checks that finding. There is no universal semantic-equivalence claim.

## Historical coverage finding

The first walkthrough failed its motif-coverage expectation: the 72-proposal schedule observed chain/collider but no fork. Preserve that result. The repaired walkthrough reports this gap and adds a separate seed-0 fork example; it does not change the 72-proposal schedule, replace a task, or hide the missing cell. Supplemental examples are excluded from default counts.

Downstream usefulness, broad generator quality and superiority remain **NOT_MEASURED**. Missingness, categoricals, subpopulation and shift processes are cataloged, not executable coverage in this slice.
