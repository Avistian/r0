# Shared-input model-tree score ensembles

`score_ensemble_recipe(members, weights=(), reduction="mean", limits=...)` compiles
frozen functional/network members into one functional recipe. This includes hard
oblique, categorical model-leaf and soft trees. All members use one shared root
and the same observed columns. Each member's deterministic internal nodes are
retained, with parent indices remapped. A final linear readout combines scalar
or class-score-vector outputs, then the ordinary task adapter runs once.

Empty `weights` means one per member. `mean` divides each weight by their finite,
nonzero sum; `sum` uses the weights directly. Signed weights support additive
score combinations. A zero-weight member is still evaluated and admitted: it
cannot hide an invalid/nonfinite function. All-zero sum weights are valid and
produce a constant signal; a zero-sum mean is rejected. Identical and correlated
members are valid, with no forced decorrelation or deduplication.

Members must share identical task controls, root nodes and root-only observed
channels. Structural noise and additional roots are rejected because moving
nodes would otherwise change their random streams. Target noise belongs to the
single final adapter. Accepted members are plain functional recipes or standalone
network recipes; fitting/calibration wrappers with additional lifecycle semantics
must not be stripped and passed off as equivalent. Frozen compiled parameters
and the effective weight matrix survive JSON and dataset archives. This compiler
does not preserve separate member construction receipts or a named reduction
choice beyond its effective coefficients; retain those receipts separately when
needed for a sampling-law audit.

This is an r0 score-composition extension, not bagging, boosting training,
probability averaging or classifier voting. A signed score sum alone does not
implement a fitted boosting algorithm. Automatic correlated-forest sampling and
full ensemble-output portfolio coverage remain open under #29.

```python
import numpy as np
from r0.modules import priors as p

task = p.LinearRecipe(rows=64, context_rows=48, features=2,
                      signal_features=2, active_features=2)
members = tuple(
    p.sample_tree_extension(
        p.TreeExtensionRequest(task, families=(family,), min_depth=1, max_depth=1),
        rng=np.random.default_rng(seed),
    ).recipe
    for family, seed in (("hard", 3), ("soft", 5))
)
recipe = p.score_ensemble_recipe(members, weights=(1.0, 3.0))
dataset = p.generate_task(recipe, seed=29, task_id=8)
```

Independent checks compare the ensemble against generated individual-member
values on exactly matched inputs for means, weighted means, signed sums and zero
sums. A three-class example compares raw scores with hand-derived matrix/bias
arithmetic. Archive reopening and generation traces are checked separately.
Aggregate node/channel and reduction-matrix admission precede compilation; final
graph admission accounts for all retained function metadata and workspace.
