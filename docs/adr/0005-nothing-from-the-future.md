# 0005 — Nothing from the future: temporal leakage rules

Every feature and every context label for a row with seed time t uses only database rows whose
time is strictly before t. Context rows at evaluation come only from the task's train table.
Normalisation is fitted on context rows only. RelBench's leaderboard rule 1 says the same, and
leakage inflates scores silently, so we enforce it in code and tests rather than in prompts.

## Considered Options
- Trust RelBench's get_db() cut-off at test_timestamp: not enough; it doesn't stop a train-split
  row from seeing rows between its seed time and test_timestamp.
- Filter per row in the featurizer (chosen): one place, testable.

## Consequences
- modules/featurize filters with `<` and has a Hypothesis property test: appending random rows
  dated at or after every seed time leaves all feature rows byte-identical.
- The evaluation harness has a spy test: a classifier's fit() sees only context rows.
- Any new feature type must extend the property test in the same PR.
