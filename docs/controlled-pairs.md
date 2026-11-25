# Controlled pairs of generated tasks

A pair contains two ordinary training tasks, one explicit intervention, and a JSON
card with both replay records, identities, checked fixed properties and measured
changes. The two variants share a root seed and task identifier. Count the pair as
one coupled root draw, with two variants, when accounting for independent tasks.

```python
from r0.modules.priors import LinearRecipe, generate_pair

pair = generate_pair(
    LinearRecipe(noise_ratio=0.2),
    intervention="noise",
    value=0.8,
    seed=27,
    task_id=9,
)
left_episode = pair.before.episode()
right_episode = pair.after.episode()
print(pair.card_json)
```

Use `paired_recipe(recipe, intervention=..., value=...)` to obtain the second
recipe before running it through the existing guarded/persistent generation path.
Use `compare_pair(before, after, intervention=...)` to verify already saved tasks.
`generate_pair` combines those operations in memory on the requested CPU/CUDA
device. It does not launch cloud work or write files. The shared explorer session
can save and reopen pairs as described below. The shared browser panel displays them
locally and in the embedded Colab explorer.

| Intervention | `value` means | Properties required to stay fixed |
| --- | --- | --- |
| `noise` | Regression noise-to-signal standard deviation ratio | Inputs, latent scores, mechanism, split and classes |
| `irrelevant` | Total feature count, at least the unchanged `signal_features` | Common input columns, latent scores, mechanism, targets, probabilities, split and classes |
| `separation` | Classification score multiplier | Inputs, latent scores, mechanism, class offsets, split and declared classes |
| `sparsity` | Active feature count | Inputs, split, classes and the unchanged recipe controls described below |

All variants also require the same seed, task identifier and execution record.
Comparison rejects additional recipe changes, different root draws, different
execution environments or broken fixed properties. A no-op request is rejected.
This is an intervention comparison between valid generated tasks; it does not
replace the existing saved-group verification and replay checks.

## Noise, nuisance inputs and classification

Existing versioned random streams separate input columns, mechanism parameters,
observation noise, labels and splits. Changing noise scales the same Gaussian noise
draw. Adding columns draws additional input streams without changing existing
columns. Classification separation changes probabilities while reusing the label
uniforms. It does not force exact class counts or make missing classes appear.
Noise applies to regression; classification uses `separation` instead.

The card reports population standard deviations over the complete observed task,
classification entropy in nats, observed/expected class counts and the fraction
of targets that changed. These are diagnostics; they are not normalization fitted
from query labels and are not model-performance scores.

## Sparsity and variance

For Gaussian linear tasks, the coefficient vector has unit squared norm per
channel. With independent standard Gaussian inputs, its expected signal variance
is exactly one before and after the sparsity intervention. Realized sample variance
can differ. The card identifies this as `analytic_unit_variance`.

Other families use an independent calibration sample to target unit signal variance.
This does **not** prove equality of their true expected variances. By default, a
nonlinear/reference/grouped sparsity pair therefore fails before generation. An
explicit `allow_estimated_variance=True` on `generate_pair` or `compare_pair`
requests a calibration-normalized comparison. Its card records
`independent_calibration_unit_variance_estimate`, preserves each normalizer and
reports realized signal spread; it never reports a proven expected variance.
The independent calibration inputs come from the original generator's keyed
stream. Normalization is not fitted to either task's observed/query targets.

Sparsity changes the active count while holding other recipe settings fixed. For
compositions, children using the implicit whole-parent mapping change their active
count with the parent; the expression, strengths, child seeds and other recipe settings
remain fixed. Explicit component mappings stay fixed. Existing recipe validation
rejects incompatible group multiples, kernel axes/scales, component mappings or
insufficient smooth terms. The API does not invent replacement geometry or drop
components to make a request valid. These exclusions are visible failures.

Reducing the active count changes the function; it need not make prediction easier
or harder for every model or draw. Compare actual model outcomes separately. The
retained walkthrough uses a fixed ordinary least-squares model with an intercept,
fits only context inputs/targets and reports query-only observed-target MSE. It
performs no tuning and makes no claim about frontier-model performance.

## Compare datasets in the explorer

Generate or open a base dataset, then use **Compare two versions of this dataset**.
Choose noise, total features, active features or classification separation and
enter the new value. The sidebar's time and output limits apply to generating
one variant on the base dataset's original device. Generation failure preserves
the previous comparison. Nonlinear sparsity requires the explicit calibration
estimate checkbox; its caveat remains visible in the saved result.

The panel plots matched original rows on identical axes. Choose a common feature,
observed targets, latent signal, regression residuals or class probabilities.
Feature labels show their before/after roles. Select a point with the mouse or
keyboard to inspect that row in both variants. These scatter plots contain observed
points with other features varying; they are not fixed-input response curves.
Measurements and class counts use all rows, including rows omitted from the plot
sample. The saved card identifies fixed properties, changed controls and variance
policy without claiming that every model must become worse.

Download both complete datasets together, download the exact comparison card or
export either chart as SVG/PNG. Reopen the saved manifest using **Open saved
comparison** in the sidebar. Creating or opening a different base clears the panel;
editing comparison controls leaves the last saved result visibly labelled with
its applied values until another comparison succeeds.

## Saved comparisons in the shared explorer session

Local and embedded Colab transports both use `ExplorerSession.request`. After
`generate` or `open` selects the base dataset, `pair` generates the variant through
the same bounded runner. It preserves the original seed, task identifier and
execution device, then checks the saved arrays with `compare_pair`. An explicit
request for a different device is rejected. Neither successful comparisons nor
failed attempts replace the base dataset. Opening or generating another base
clears the selected comparison.

```python
from pathlib import Path

from r0.modules.explorer import ExplorerSession

session = ExplorerSession(Path("/tmp/r0-pair-example"))
base = session.request("generate", {"recipe": {"noise_ratio": 0.2}, "seed": "27", "task_id": "9"})
comparison = session.request(
    "pair",
    {
        "intervention": "noise",
        "value": 0.8,
        "expected_episode_id": base["view"]["episode_id"],
        "max_seconds": 30,
        "max_bytes": 1048576,
    },
)
reopened = ExplorerSession(session.output_dir).request(
    "open_pair", {"path": comparison["pair"]["path"]}
)
export = session.request(
    "pair_export",
    {"expected_pair_id": comparison["pair"]["card"]["pair_id"], "max_bytes": 2097152},
)
```

`pair` returns the unchanged base view and a `pair` object with the second view,
archive location, run result, comparison card and saved manifest path. Browser
records encode large seed/task/split integers as decimal strings; `card_json`
and each `recipe_json` retain the original integers for lossless download/replay.
The stable sampled row indices match across both views because they share the seed,
task identifier and row count.

The manifest under `pairs/` links two ordinary runner archives. `open_pair` verifies
both archive receipts and recomputes the comparison card before changing the
session. All paths must resolve inside the output directory. Moving the entire
output directory preserves its relative links. These are local generated artifacts,
not an upload interface. Run records retain the original paths and measurements.

Generation's byte/time limits apply to the new variant's runner request; existing
base data and diagnostic attempt/manifest records are not charged to that archive
limit. Each comparison manifest is limited to 1 MiB. A rejected post-generation
comparison can leave a valid variant archive: `.pair-attempts/` records the failure
separately from the worker's successful generation. For example, estimated-variance
permission is checked by the shared comparison function after the variant is saved.
No failed comparison is returned as accepted or counted as a second independent draw.

`pair_export` returns base64 ZIP bytes under one total `max_bytes` limit. It contains
`card.json`, `before.zip` and `after.zip`. Each nested archive has its complete
recipe, mechanism, identities, predictor arrays/CSV and separate diagnostic truth
arrays/CSV. Query targets remain outside predictor data. Exports contain all rows;
only the returned chart views are sampled. The optional `expected_episode_id` and
`expected_pair_id` guards reject requests from stale views.

## Replay and remaining delivery

Both task replay records and mechanism fingerprints are retained in the card.
Regenerate each side with the recorded recipe, seed, task identifier and execution
environment. The ordinary task objects retain their complete hidden parameters;
model consumption continues through their unchanged public `episode()` method.
Pair failures are explicit; this API does not retry, replace tasks or silently
relax the variance policy. Persistent callers retain attempts using the existing
runner rather than treating a failed pair as two completed independent tasks.

CPU API and local browser coverage do not establish GPU agreement or hosted Colab
interaction. Those checks, broad sampling recipes, full resource coverage and the
remaining #27 integration work are tracked separately.
