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

## Generated graph archives

`sample_parameterized_xgboost_graph(skeleton, node_priors, parameter_rng=...,
input_rng=..., target_rng=..., backend_rng=...)` constructs an authored functional
DAG with sampled independent XGBoost fits. `node_priors` is an ordered tuple of
`(node_index, XGBoostParameterPrior)` pairs; selected nodes must retain their input
and output dimensions and have no pre-fitted numerical pipeline. Root and internal
mechanisms are supported. Multiple fits retain the shared parameter-stream chain.

The result is a `SampledXGBoostGraph`, version
`sampled-xgboost-functional-causal-v1`. Its archive binds every parameter sample to
the exact fitted plan and retains the raw parameter draws, fit draws, native model
and exported function. Ordinary `generate_task`, JSON decoding, task-group
save/reload, the generation CLI and numerical traces preserve this record. Replay
uses frozen functions without importing the fitting backend. Explicit root-stage
interventions retain the original parameter samples and fits. Construction failures
roll back all four caller RNG streams.

The graph admits aggregate parameter/fit storage and replay workspace, including
metadata and worst-prior fitting estimates. The latter is conservative even when
replay does not refit. Exact admitted-memory-boundary reconstruction and rejection
one byte below that boundary are checked before nested RNG restoration.

Five integration checks cover generated task ZIP/replay, large task IDs, repeated-row
traces, memory boundaries, aggregate two-fit preflight, parameter-stream chains,
backend failure rollback, backend-free replay and a root intervention. Combined with
existing fit-graph and parameter checks, 21 tests pass. An actual CLI run also created
two persisted datasets from the complete sampled graph JSON. Evidence is ignored
under `docs/evidence/issue-28-sampled-xgboost-graph/`; its first trace-dispatch failure
is retained alongside the repair.

This integration uses independent synthetic fitting. Existing support/batch/ancestor
XGBoost population lifecycles remain separately implemented; automatic sampled-plan
composition with those population workflows is not claimed here. The entire source
prior and all option-level evidence are not closed by this component. Independent
review and current-head full verification remain separate.
