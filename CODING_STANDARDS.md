## Every random draw takes an explicit Generator
Pass `rng: np.random.Generator` (or a torch Generator) in; never call global `np.random.*`.
Why: a run that can't be replayed can't be debugged or compared.

## "Strictly before" is spelled `<`, and it is tested
Any filter on a time column uses `time < seed_time`, never `<=`, and comes with a test that
adds rows at exactly the seed time and shows nothing changes. Why: off-by-one leakage is the
most common silent bug in relational ML.

## Normalisation statistics come from the context only
Means, standard deviations, category maps: fit on context rows, apply to queries.
Why: fitting on queries leaks their distribution into the prediction.

## A metric is computed on query rows only
Context rows never contribute to a reported number.

## A result-changing config edit needs a new run record
Change a config that alters results → new run id, new record. Never overwrite one.

## Completion claims distinguish choices from missing evidence
For protocol/specification reviews, classify pending work as an unresolved decision, a fact
awaiting verification, or an authorization. Name its owner and the stage it blocks. Reviewers
check that completion language matches those distinctions; accepted scope can still defer
implementation choices. See [protocol review](docs/agents/protocol-review.md).

## Verification claims have a retained reproduction path
Reviewers require identified inputs and the exact command or retained program needed to
reproduce claimed calculations/checks in the declared environment. Compare what it actually
checks with the claim; a printed PASS or another reviewer's independent calculation is not
itself a reproduction recipe. Keep unsupported portions explicitly unverified. Hand calculations
remain valid evidence when their inputs and derivation are retained; executable-check claims
also need the executed procedure. See the [worked review example](https://github.com/Avistian/r0/blob/0e4801bde17896383862d0d695ae81334ebb7aee/docs/evidence/issue-25-retro/README.md).
