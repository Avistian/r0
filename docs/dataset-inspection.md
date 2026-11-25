# Dataset inspection and exports

This is the shared data layer for the local and Colab explorer in #27. The
browser interface, generation graph and notebook integration are separate slices.

```python
from pathlib import Path
from r0.modules.priors import LinearRecipe, generate_task
from r0.modules.explorer import inspect_task, inspect_rows, replay_bundle

task = generate_task(LinearRecipe(rows=513, context_rows=300), seed=27, task_id=5)
view = inspect_task(task, sample_rows=17)
row = inspect_rows(task, [512])[0]
Path("dataset.zip").write_bytes(replay_bundle(task))
```

Every summary and histogram uses all generated rows. Expensive chart and table
views use a uniform sample without replacement, keyed by seed, task identifier
and total row count. Changing noise or adding irrelevant columns preserves this
selection. `shown_rows` and `total_rows` make the difference explicit. Both chart
sampling and individual row inspection accept at most 2,048 rows. Inspection
does not sample a new mechanism or change data.

Feature summaries identify active and irrelevant columns. Regression exposes
latent signals and realized observation noise. Classification retains every
declared class, including those absent from context or the whole task, and reports
observed counts, expected counts, entropy and maximum probability. Entropy uses
natural logarithms. These are diagnostics; they do not filter tasks.

Statistics scale finite values before computing mean, standard deviation,
quantiles and histogram edges, so large finite input scales do not overflow
intermediate squares. An unrepresentable mean or standard deviation is `null`.
Constant histograms have one bin and equal edges. No normalization is fitted for
model consumption.

The ZIP replay bundle contains the full dataset, independently of chart sampling:

| File | Contents |
| --- | --- |
| `recipe.json` | Versioned recipe, seed, task identifier and execution record |
| `identity.json` | Source and episode identities, task type and class names |
| `mechanism.json` | Sampled parameters and normalization |
| `predictor.npz` | Context inputs/labels, query inputs, original row indices |
| `truth.npz` | All targets, latent scores and classification probabilities |
| `predictor.csv` | Inputs and split membership; query labels are blank |
| `truth.csv` | Targets, scores and probabilities by original row index |

NumPy arrays are lossless and require no pickle. CSV uses round-trippable numeric
representations, with integer class labels. Diagnostic truth is intentionally
separate from predictor files. Only share the predictor files with a model.
The archive is uncompressed. `max_bytes` defaults to 256 MiB and limits output
storage; exceeding it raises an error. It is not a process-memory limit: arrays
and the archive are built in memory. Bounded bulk generation uses the existing
generation runner and persistent groups instead.

Focused verification: `python3 scripts/run_check.py pytest tests/explorer -q`.
The tests inspect full versus sampled summaries, absent classes, original row
identities, unmodified tensors, large input scales, complete exports and explicit
storage-limit failure. They do not establish browser or hosted Colab acceptance.
