# Issue 21 episode contract

Status: implemented locally. Local acceptance artifacts are in the ignored `docs/evidence/issue-21/` directory. Integration review remains pending.

## Accepted decisions

- An episode is one context/query prediction problem drawn from a source task. Multiple episodes may share a task; preserve both identities for overlap auditing.
- The predictor receives context features and labels, query features and declared inference-available task information. Query targets and hidden mechanism details are held in a separate scoring/diagnostic record. This is an interface boundary, not a security sandbox.
- Preserve continuous, integer/count, binary, unordered categorical, ordered categorical and timestamp feature meanings with explicit missingness. Unsupported feature kinds produce clear errors. Free text, nested values and embeddings remain explicit future extensions. Distribution shape, correlations and latent mechanisms belong to diagnostics.
- Record generator/configuration version, root seed, source-task identity and episode identity. Exact replay is scoped to a pinned software environment; requesting an episode is independent of generation order and worker assignment.
- Use context-target mean for the regression demonstration and context-class frequencies for classification. Inspect configuration, preprocessing, predictions and query-only scores end to end.

## Accepted decisions from round 2

- Hidden mechanisms, corruption events, relevance flags and latent groups stay outside the predictor view unless an experiment explicitly declares them observable inputs.
- Each episode owns its schema and dimensions. Context and queries share that schema. Batch padding is introduced later and is distinct from missing values.
- Category mappings fit context only. Unknown query categories and missing values are distinct. Preserve raw categories for inspection. Distinguish all-missing context columns from constant observed columns.
- Classification vocabulary is declared independently of query targets, with recorded provenance. Context may omit declared classes. The frequency fixture gives absent classes zero probability. Reject labels outside the declared vocabulary.

## Accepted decisions from round 3

- Preprocessing is explicit and versioned. Preserve the raw episode and create a separate transformed view with fitted context statistics and maps.
- Initial fixture recipe standardizes observed continuous/count context values with population standard deviation and scale one for constant columns. Preserve binary values. Unordered categories use context-fitted mappings; ordered categories require declared order. Timestamps preserve declared units/time convention. Calendar/cyclic engineering remains a later adapter decision.
- Preserve missingness, pair placeholders with masks and identify all-missing context columns. Regression fixture targets retain original units. Record the recipe identity without choosing a production default.
- Require at least one context row, query row and feature. Scored fixture targets must be complete and finite. Reject overlapping/duplicate row identities, inconsistent schemas, infinite numeric inputs, invalid categories and missing scored targets. Repeated feature values across different rows are valid.
- Configurations are strict and versioned. Reject unknown keys/versions, invalid values and contradictions. Persist resolved defaults. Identity is based on semantic configuration content. Generation and splitting randomness are independently reproducible.
- Provide a small replay command with identities, resolved configuration, raw rows, fitted transformations, exact predictor view, predictions and separate scoring view. Use MSE for regression and multiclass Brier score for classification; benchmark selection remains in issue 25.

## Module ownership and entry point

- Introduce a low-level episodes module owning schema, identities, validation and predictor/scoring views. Update import boundaries so generators, preprocessing, batching and models can use its public interface without dependency cycles.
- Keep generation in priors and context-fitted transformations in featurize. Use existing dependencies.
- Introduce replay as the owner of fixture orchestration, simple predictors, scoring and report writing. Expose r0 replay --config … --out … through the existing CLI parsing and quiet/error policy. Consume public module entry points. Benchmark harness development remains in issue 26.

## Replay and ownership guarantees

- Exact semantic replay within the recorded environment includes schema, row identities, values, masks, partitions, fitted transformations and predictions. Record environment versions.
- Record resolved configuration identity separately from the episode-content fingerprint. Timestamps and output paths do not affect those identities.
- The predictor input owns its data independently. It must not provide indirect access to scoring labels or allow accidental mutation of the retained raw episode.

## Acceptance walkthrough

Provide tiny regression, binary and three-class fixtures. Across them cover each declared feature kind, unknown categories, missingness, constant and all-missing context columns, and a class absent from context.

A reviewer follows one row from the source fixture through its fitted transformation, predictor view, prediction and query-only score. Retain worked calculations next to the report.

1. Recompute a transformed numeric value using observed context values and population variance. Constant columns use scale one.
2. Recompute MSE averaged over query rows and multiclass Brier score summed over the declared classes then averaged over query rows.
3. Change query targets within the declared task domain. Predictor inputs and predictions remain identical while scoring may change.
4. Request the same episodes in a different order and recover identical semantic contents.
5. Change only the split seed and recover the same source task with the declared alternative partition.
6. Inspect actionable errors for an invalid configuration and overlapping row identity.

Preserve the config, recorded environment, inspectable reports and reviewer observations as acceptance evidence. Repository checks still apply to implementation. No production model, benchmark protocol or production generator recipe is selected by this fixture work.
