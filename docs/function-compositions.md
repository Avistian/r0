# Composed analytic functions

`CompositeRecipe` combines saved analytic functions on the same observations. Its
`task` controls the final table, split and targets. `inputs` controls independent
feature draws. `FunctionComponent` supplies a standalone recipe, an optional feature
subset and a nonzero signed strength. `Combination` supplies `sum` or `product`,
children and another nonzero signed strength. Integer children reference components
by index; reusing an index reuses the same sampled function.

```python
from r0.modules.priors import (
    Combination,
    CompositeRecipe,
    FunctionComponent,
    PolynomialRecipe,
    SmoothRecipe,
    generate_task,
)

recipe = CompositeRecipe(
    components=(
        FunctionComponent(PolynomialRecipe(degree=2)),
        FunctionComponent(SmoothRecipe(family="sinusoidal"), strength=0.5),
    ),
    expression=Combination("product", (Combination("sum", (0, 1)), 0)),
)
task = generate_task(recipe, seed=27, task_id=4)
```

There are 1–64 distinct components, at most 64 component occurrences in the expression,
and at most eight nested combination nodes on any path. Each combination has 2–64
children. Every declared component must be used. The expression's interaction order
is the maximum number of multiplied leaf occurrences in a term after distributing
products over sums. A repeated leaf contributes again to this order.

## Inputs, sparsity and targets

The parent samples its declared active feature set once. Component `axes` index that
set; empty means all parent active features. Their union must cover the set. Each
component's active count must equal its selected axis count. Its saved active columns
are mapped, in their recorded order, onto those parent columns. Other child columns
are zero during evaluation because the sampled function does not depend on them.
This is coordinate relabeling, not a new random draw. Grouped components retain their
selected group membership under the recorded mapping; mapped groups need not be
adjacent parent columns.

Components sample independent mechanisms using keyed child seeds. Their standalone
observations/targets initialize the existing sampler but do not become parent
observations or training examples. Cards retain the child mechanism, seed, feature
count and parent-column mapping. The final result is one independently sampled source
task, regardless of the number of components or repeated references. Child split
and context-size controls are retained for replay but excluded from the parent
source identity, just as the parent split/context controls are excluded.

The parent and components must use the same input distribution and score-output type.
Component observation noise, separation and offsets must remain neutral; set these
on the parent. Linear-v1 and Fourier-v1 have fixed standard Gaussian inputs; other
analytic recipes expose `InputDistribution`. For a non-Gaussian linear component,
use degree-one `PolynomialRecipe` or `GroupedLinearRecipe` with group size one.
Direct GP draws are reference samplers and are explicitly excluded from analytic
function compositions; their covariance sums/products remain separately supported.

Parent noise and nuisance-column pairs preserve child mechanisms, active inputs and
splits. `evaluate_signal` evaluates the whole saved expression on new input rows.
The output remains latent diagnostic truth; the common target adapter generates
regression observations or categorical labels afterward.

## Numerical limits and normalization

The final expression has its own calibration on independently drawn shared inputs, using
256–16,384 calibration rows. Leaves retain their own recorded calibration. A CPU
calibration pass is shared across CPU/GPU runs; requested-device evaluation operates
in 256-row chunks. The estimate is not a proof of exact population unit variance.

For unbounded Student-t inputs, a conservative growth-degree calculation requires
`df > 2*growth_degree`: linear contributes one, polynomial its configured degree,
quadratic two, bounded trigonometric/radial functions zero. Sums take the maximum;
products add degrees, including repeated leaves. Bounded input distributions avoid
this moment restriction. This check guarantees finite required moments, not that
any small calibration sample estimates a heavy-tailed variance accurately.

Zero/nonfinite calibration scale, nonfinite arithmetic, or a product rounded to zero
from nonzero operands fails visibly. Expression order is recorded. Products are not
symbolically rearranged to rescue underflow or overflow; no component, strength or
failed task is silently removed. Supported limits are not a promise that every
Cartesian combination of sizes, degrees and strengths is numerically valid or
qualified for bulk generation.

## Run locally

```sh
python3 scripts/run_check.py python scripts/generate.py \
  --config configs/generation/composite-cpu.toml --out /tmp/r0-composite-run
```

The ordinary guarded runner handles grouped persistence, limits, worker execution,
failed identities and restart verification. Graph rendering, broad sampled exploration
recipes, hosted checks and repeated throughput measurements remain separate #27 work.
