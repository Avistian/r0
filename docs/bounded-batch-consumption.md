# Consume generated episodes in bounded batches

From the repository root, using the environment described in the README:

```sh
python scripts/run_check.py python scripts/consume_generated_batches.py --out /tmp/r0-batches
```

This small CPU example generates 30 independent tasks: ten families across
regression, binary and multiclass classification. Rows and feature counts vary;
one task declares 100 classes, including classes absent from context. It takes
at most four episodes at a time and performs one optimizer step per batch,
including the final partial batch. `--count` changes the task count (up to 300);
`--batch-size` changes the bound (up to eight). Every generated task uses the
explicit root `--seed` and a different task identifier.

To consume a saved local generation group instead:

```sh
python scripts/run_check.py python scripts/consume_generated_batches.py \
  --archive /path/to/group.zip --count 30 --out /tmp/r0-saved-batches
```

The saved source stops at the requested count or archive exhaustion, whichever
comes first. It verifies the group receipt and loads each task without generation.
The summary distinguishes requested and consumed counts and records the archive
hash. Archive receipt verification reads the complete bounded group; it does not
load all task arrays together.

## What reaches the optimizer

```text
shared generate_task(...) or load_group(...)
  → TrainingTask.episode()
  → up to batch_size episodes
      → predictor_view() → context-fitted features → predictions
      → query targets → loss
  → mean episode loss → backward() → one SGD step
```

The predictor learns one shared log-bandwidth scalar. It weights context rows by
squared standardized distance, then averages their labels. Regression uses query
mean squared error. Classification uses query negative log probability, with a
fixed 1% uniform probability mass to keep absent context classes legal. Each
episode has equal weight in the batch objective. Changing query labels leaves
predictions exactly unchanged; the script checks this for every episode.
Generator scores, probabilities and sampled mechanisms never enter the predictor.
Neither predictor nor optimizer dispatches by generator family.

The sweep demonstrates bounded consumption through the common episode contract.
It is not the production mixture or training stream, which remain #30, nor the
foundation model or `r0 train` implementation. Mixed regression/classification
losses have different units; their aggregate is an execution check, not a model
quality metric. This predictor deliberately supports only complete numeric
features, at most 128 rows, 256 features and 100 classes per task. These are example
limits, not generator limits.

## Retained evidence and independent check

Every run uses a new output directory. It saves source revision and script hash,
software versions, requested and actual counts, per-query predictions, per-episode
losses, batch gradients and parameter updates. Failures after setup retain the
current task and error without retries. The generator holds no task history;
the consumer releases each batch before requesting the next. Retained output on
disk grows with task count. Reported process peak RSS includes Python/Torch and
allocation caches, not just live batch tensors.

```sh
python scripts/run_check.py python scripts/verify_generated_batches.py \
  /tmp/r0-batches --output /tmp/r0-batches-verified.json
```

The verifier replays recorded inputs or reads the same saved group, independently
computes NumPy distance weights and query losses, and compares central finite
differences to Torch gradients. It checks every parameter update, task identity,
batch count, shape and final summary. It never calls the consumer's predictor or
feature transformer. The counterfactual claim comes from the consumer's explicit
check; the verifier checks its recorded status rather than independently repeating
that intervention.

`scripts/saved_batch_walkthrough.py --out /tmp/r0-saved-demo` constructs seven
small tasks and consumes their saved group with `generate_task` disabled during
consumption. Verify `/tmp/r0-saved-demo/consumption` with the same verifier.
All commands should run with fixed thread counts when comparing resource records.
