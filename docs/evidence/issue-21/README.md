# Episode walkthrough

Run from the repository root in the project environment.

    uv run r0 replay --config configs/fixtures/regression.toml --out /tmp/regression.json
    uv run r0 replay --config configs/fixtures/binary.toml --out /tmp/binary.json
    uv run r0 replay --config configs/fixtures/multiclass.toml --out /tmp/multiclass.json
    uv run python scripts/episode_walkthrough.py

The three adjacent JSON files retain raw source tasks, partitions, resolved configuration, environment versions, fitted statistics, the predictor-only view and the separate scoring view. The CLI reports are deterministic for the recorded environment. Source mechanism details are diagnostic and never passed to the fixture predictor.

## Hand calculations

Regression context rows 3, 1, 2 have targets 7, 3, 5. Their mean is 5. Query targets 1, 11, 9 yield squared errors 16, 36, 16 and MSE 68/3. Replacing query targets with zero preserves inputs/predictions and yields MSE 25.

For regression measurement, context values 9, 5, 7 have mean 7 and population variance 8/3. Query row 0 has raw value 3, so its transformed value is (3−7)/sqrt(8/3), approximately −2.449489743. The query values do not enter the fitted mean or variance.

Binary context labels B, A, A produce probabilities (2/3, 1/3) in declared order (A, B). Query labels B, B, A yield Brier terms 8/9, 8/9, 2/9, averaging 2/3.

Multiclass context A, B, B produces probabilities (1/3, 2/3, 0) in declared order (A, B, C). Query labels C, A, A yield terms 14/9, 8/9, 8/9, averaging 10/9. The class vocabulary comes from the task design, independently of query labels.

The categorical map is blue=0, red=1 from context. Query green receives value −1 with unknown=true and missing=false. An absent color receives placeholder 0 with missing=true and unknown=false. Constant numeric columns use scale one; unobserved columns record all_missing=true, mean zero and scale one. Missing placeholders are accompanied by masks.

Ordered categories use the declared intrinsic order low, medium, high. Timestamps are numeric Unix epoch values with explicit UTC and s/ms/us/ns units. This fixture supports integer counts as nonnegative values; general signed integers can use continuous representation until a distinct type is introduced. Timestamps remain features, and no relational history filtering is introduced by this single-table transformation.

## Identity and failure evidence

The retained walkthrough shows target replacement, reordered requests and split changes. Task generation and splitting use separate explicitly created NumPy generators derived from stable keys. Changing the split keeps the source ID and source data fixed. Configuration fingerprints resolve defaults; content fingerprints include raw episode values, transformed values and predictions. Report timestamps and output paths are excluded.

Invalid config keys and overlapping identities produce explicit errors. The report writer serializes fully before atomically replacing its destination. Frozen dataclasses and nested immutable tuples prevent normal caller mutation of raw/predictor data. This is an API ownership guarantee, not a sandbox.

The initial episode, transformation and replay interface checks failed before implementation; the CLI replay case failed before wiring. A later malformed-prediction check exposed non-finite scoring and was corrected. Independent review and final repository checks are recorded in review.md.
