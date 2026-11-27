# Independent variance matching for network input views

This library slice adds an explicit r0 extension for comparing two input masks of one frozen network. It estimates each output channel's mean and population standard deviation on one shared, independent root population, then transforms only the after-mask scores:

`matched = (after - estimated_after_mean) / estimated_after_std * estimated_before_std + estimated_before_mean`.

This matches estimated baseline moments, not unit variance, exact population moments, covariance, class frequencies, or empirical moments of a generated episode. It does not reproduce an undisclosed paper normalization law. Classification matching applies to raw score channels before the existing task link, separation, offsets and label sampling.

## Public seam

```python
matching = fit_network_variance_match(
    original_network_recipe,
    before_inputs=(0, 1),
    after_inputs=(1,),
    rows=256,
    rng=independent_rng,
)
after = MatchedNetworkInputView(
    original_network_recipe,
    after_task,
    (1,),
    matching=matching,
)
```

The supplied RNG is owned by this construction and advances atomically only on success. The fit API accepts no query inputs, episode labels or task seed. Callers must supply a separately owned calibration RNG. The paired-comparison layer can derive this RNG from original archive identity, both masks and a protocol tag; it must not derive it from episode/query identities. The public fit API accepts 2–4096 calibration rows and an original `NetworkRecipe`, not an already calibrated view. Further matching of calibrated views requires a separate protocol and remains unsupported here.

The record retains the unchanged original recipe, both masks, root population, both raw score tables, moments, RNG snapshots, exact raw variates and a population/score digest. Root covariance and ball transformations happen before either mask, and both masks use precisely the same population. Restoring the record replays the root draws and both frozen functions and verifies every retained result. No backend fitting occurs. Zero or nonfinite dispersion in either mask, overflow, invalid masks and exceeded limits fail without retries or fallback scales. Aggregate preallocation includes nested function workspace and row floors, retained population/scores, duplicated source metadata and raw custody.

The new `network-input-view-v2` archive binds the calibration to the original base and after mask. The original network and `network-input-view-v1` records remain unchanged. Generation and `evaluate_signal` apply the frozen affine transform; changing episode rows, seed or task identity does not refit it. The mechanism labels estimated matching explicitly.

## Scope and verification

Author checks cover independent NumPy arithmetic for regression and multiclass scores, Gaussian/uniform/ball/covariance roots, retained RNG state, strict archive tamper rejection, pre-draw cap rejection, failure rollback, and legacy archive/output hashes. Evidence is under ignored `docs/evidence/issue-28-network-variance-match/`.

Paired-comparison integration, inspection and explorer controls are separate required follow-ons. This slice adds no UI and does not claim the entire feature-extension/sparsity scope is complete. Full-suite integration, independent review, hosted runs and GPU checks are not supplied by these author checks.
