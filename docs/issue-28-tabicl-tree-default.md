# TabICL noncausal tree configuration

`sample_tabicl_tree_graph` builds a sampled XGBoost graph with the pinned
noncausal architecture. It samples one or two hidden stages uniformly, one shared
hidden width from 3 through 9 uniformly, and adds an output tree stage. Original
Gaussian roots are the observed features; final-stage channels are the target.
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
