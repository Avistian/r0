# Sampled XGBoost depth and tree-count laws

`XGBoostParameterPrior` and `sample_xgboost_plan` implement the recorded
TabICL depth/count laws for actual `XGBoostFitPlan` records. This addresses the
parameter-sampling part of `I1.function.07`; it does not close the entire source
recipe. The [source-law audit](research/0028-fitted-tree-source-laws.md) retains
the paper/code distinctions and pinned source identity.

The `tabicl-pinned-depth-count-v1` profile draws independently, in order:

```text
D = min(4, 2 + floor(Exponential(scale=2)))
T = min(4, 1 + floor(Exponential(scale=2)))
```

The terminal bin retains the entire exponential tail. Raw unbounded draws, initial
and final RNG snapshots, the prior, realized plan and source pin are retained in
`SampledXGBoostPlan`. Decoding replays the draws and rejects changed plans or source
claims. Sampling pre-admits the largest fitting plan plus parameter metadata before
copying/drawing from the caller stream. Failed transactions preserve the caller RNG.
The pinned profile requires regression, matching the source's artificial Gaussian
vector targets; it is not a classifier hyperprior.

For broader explicit exploration, `r0-capped-exponential-xgboost-v1` allows positive
finite depth/count rates, depth caps 2–8 and tree-count caps 1–32, subject to existing
fit resource limits. It retains offsets two and one. These are named r0 exploration
bounds, not assertions about undisclosed source priors. The base plan retains the
remaining histogram/objective/fitting controls; this sampler does not change its
fitting lifecycle or claim historical backend-default equivalence.

```python
import numpy as np
from r0.modules import priors as p

sample = p.sample_xgboost_plan(
    p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=2, output_width=2, rows=32)),
    rng=np.random.default_rng(28),
)
fit = p.fit_xgboost(
    sample.plan,
    input_rng=np.random.default_rng(29),
    target_rng=np.random.default_rng(30),
    backend_rng=np.random.default_rng(31),
)
```

The parameter sample and fitted record are separately serializable using
`dataclasses.asdict`; their decoders are `sampled_xgboost_plan_from_dict` and
`xgboost_fit_from_dict`. Keep both records: the fitted record alone retains the
realized settings but does not retain their sampling prior.

Nine focused checks cover independent NumPy expressions and exact stream state
for all five RNG engines, raw tail retention, source/plan tampering, pre-admission
before snapshot restoration, and actual XGBoost fitting/export/reopen. Together
with existing fitting checks, 32 tests pass. Raw evidence, including the initial
missing-API failure and repaired ledger-transaction failure, is ignored under
`docs/evidence/issue-28-xgboost-parameter-laws/`.

Automatic graph construction and task archives retaining these parameter samples
alongside their fitted mechanisms remain unfinished. This is a sampler component,
not completion of `I1.function.07`, a full source prior, or current-head full-suite
acceptance. Independent review and the complete repository check remain separate.
