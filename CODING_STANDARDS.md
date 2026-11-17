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
