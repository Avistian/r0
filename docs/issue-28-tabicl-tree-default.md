# TabICL noncausal tree configuration

`sample_tabicl_tree_graph` builds a sampled XGBoost graph with the pinned
noncausal architecture. It samples one or two hidden stages uniformly, one shared
hidden width from 3 through 9 uniformly, and adds an output tree stage. Original
Gaussian roots are the default observed features; final-stage channels are the target.
The first tree stage has no structural noise. Later stages use standard deviation
0.01 by default, configurable with `noise_std`. With `pre_sample_noise_std=True`,
channel scales are frozen absolute Gaussian draws. Depth and estimator counts
use the existing retained capped-exponential sampler independently at every stage.

The architecture/noise rules follow the
[pinned tree implementation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py#L193-L269).
This public helper deliberately names its independent-fit adaptation: each tree
fits an independent Gaussian table and Gaussian vector targets once, using
explicit bounded backend settings and single-threaded fitting. It does **not**
fit each layer's actual incoming batch. Actual-parent population fitting remains
available through its separate [lifecycle API](issue-28-xgboost-population.md);
automatic population-policy compositions remain tracked in #126. The helper
also does not reproduce the complete root/hyperparameter mixture or unspecified
historical backend defaults. It uses the existing XGBoost defaults in its fit
plan, not an assertion about the upstream installed version.

The output is the existing `SampledXGBoostGraph`: sampled depth/count records,
actual fits, noise scales, graph geometry and selected observations survive
recipe JSON, generated dataset archives and numerical tracing. The architecture
is retained as realized geometry; no additional architecture sampler provenance
record is introduced. All five caller streams are committed together only after
successful construction and fitting.

A small usable example:

```python
from pathlib import Path

import numpy as np
from r0.modules import priors, generation

task = priors.LinearRecipe(
    rows=64, context_rows=48, features=2, signal_features=2, active_features=2
)
names = ("architecture_rng", "parameter_rng", "input_rng", "target_rng", "backend_rng")
streams = {name: np.random.default_rng(28 + i) for i, name in enumerate(names)}
recipe = priors.sample_tabicl_tree_graph(task, **streams, fit_rows=32)
sample = priors.generate_task(recipe, seed=28, task_id=0)
generation.save_group(Path("tabicl-tree.zip"), [(0, sample)], max_bytes=8 * 1024**2)
```

Focused checks exercise both hidden-depth outcomes, fixed and sampled noise,
root/final readout selection, actual fitting, JSON/ZIP replay, repeated-row
traces, admission rejection and all-stream rollback after backend failure.
Full-suite acceptance of this newer helper remains pending.

## Authored root extensions

The optional `root` argument accepts the existing joint, coordinate-law,
sampled-coordinate-law and frozen-prototype root records. Width must match the
task's features. Uniform-box and uniform-ball roots, mixed coordinate laws,
independently calibrated sampled roots and prototype mixtures therefore compose
with the same source-shaped architecture. This is configurable r0 breadth, not
an assertion that the complete upstream root-selection mixture is reproduced.

Root selection does not consume any architecture or fit stream. Fitting continues
to use independent Gaussian input/target tables; changing emitted roots does not
refit on the changed input distribution. A root can consequently produce inputs
outside a tree's fitting range and constant or weak outputs remain valid. Existing
root admission, calibration provenance and replay limits apply unchanged.
Source categorical coordinate values remain numerical graph inputs here; this
helper does not declare native categorical feature schemas. Use typed observation
encodings or the native graph path when categorical schema semantics are needed.

For example, pass this root to the helper in the example above:

```python
root = priors.SourceRootBase(
    priors.SourceRootLaw(
        (
            priors.SourceRootCoordinate("unit_uniform"),
            priors.SourceRootCoordinate("categorical", probabilities=(0.2, 0.8)),
        )
    )
)
```

The realized graph can be serialized in a `generation.RunRequest` and consumed by
`r0 generate`. Focused checks cover five root cases, unchanged fit records,
JSON/archive replay, repeated-row tracing and rejection without stream consumption.
A retained CLI example generated two datasets with uniform and binary-coded root
coordinates. Logs and the request-construction script are gitignored under
`docs/evidence/issue-28-tabicl-roots/`. An initial test used 16 fit rows, below the
existing 32-row minimum; that failed fixture and the corrected run remain retained.
