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
device. It does not launch cloud work or write files. The explorer comparison
panel and persistent pair exports are the next #27 integration slice.

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

## Replay and remaining delivery

Both task replay records and mechanism fingerprints are retained in the card.
Regenerate each side with the recorded recipe, seed, task identifier and execution
environment. The ordinary task objects retain their complete hidden parameters;
model consumption continues through their unchanged public `episode()` method.
Pair failures are explicit; this API does not retry, replace tasks or silently
relax the variance policy. Persistent callers retain attempts using the existing
runner rather than treating a failed pair as two completed independent tasks.

CPU API coverage does not establish GPU agreement, hosted Colab interaction or
comparison UI acceptance. Those checks, broad sampling recipes, full resource
coverage and the remaining #27 integration work are tracked separately.
