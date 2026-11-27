# Issue 28 atomic delivery mapping

This authored index connects individual inventory IDs to implementation contracts.
It supplements, rather than replaces, the complete [723-record inventory](research/0028-source-option-inventory.md).
**Unlisted IDs are NOT_AUDITED by this delivery index.** A source-family name,
passing suite, or shared mathematical operator does not close an unlisted ID.
No whole-issue completion percentage is inferred from these rows.

`SCOPED` means that the named component exists with bounded independent checks;
it does not establish its entire paper's sampling law, every composition, hosted
execution, resource envelope or predictive quality. `PARTIAL` identifies an
explicit semantic/lifecycle gap. `PENDING` identifies missing implementation.
Raw review records and worked examples are retained in ignored `docs/evidence/`;
[retention limits](issue-28-evidence-retention.md) apply to fresh clones.

## Parent composition

Common delivery contract: `ParentCompositionPrior` / `FrozenParentFunction`,
`sampled-parent-graph-v1`, `geometry-parent-request-v1` and
`native-parent-graph-request-v1`. Parent order and individual input widths are
retained; native members consume encoded state widths and produce score widths.
See [equations and source adaptations](research/0028-parent-composition.md) and
[geometry/pipeline/native integration](research/0028-parent-integration.md).

| Inventory ID | Named component | State and qualification |
| --- | --- | --- |
| I2.node.06 | Concatenate parents before one frozen function | SCOPED; explicit `concat` mode retains joint-input semantics. |
| I2.node.07 | Separate parent functions | SCOPED; disjoint ordered slices and independently fitted children. |
| I2.node.08 | Sum reduction | SCOPED; elementwise member sum. |
| I2.node.09 | Product reduction | SCOPED; elementwise member product with dependent-moment bounds. |
| I2.node.10 | Maximum reduction | SCOPED; elementwise member maximum. |
| I2.node.11 | Logsumexp reduction | SCOPED; numerically stable elementwise reduction. |
| I2.node.13 | Aggregation-only selector | SCOPED; forced aggregate mode with source-specific operator catalog. |
| M2.mechanism.08 | Independent linear parent projections | SCOPED; explicit matrix-law adaptation, not an undisclosed source-law reproduction. |
| M2.mechanism.09 | Joint-member mean | SCOPED; joint function participates in the same mean as projections. |
| M2.mechanism.10 | Joint-member weighted reduction | SCOPED; Gaussian-logit softmax weights are a named r0 adaptation. |
| M2.mechanism.11 | Joint-member neural reduction | SCOPED; member-major concatenation and independent reducer fitting; parameter law adapted. |
| M2.mechanism.12 | Joint-member product | SCOPED; joint result is a product member, not an added residual. |
| M2.mechanism.13 | Joint-member maximum | SCOPED; maximum includes the joint result and projections. |

Public entry points are exported by
[the parent facade](../src/r0/modules/priors/lib/parent_api.py).
Independent review of `8097c3b` covered arithmetic, raw custody and 14 fit
populations; follow-on `365a9a5` covered 30 native fit populations, archive
mutations, transaction rollback and targeted legacy byte identity. Integration
`723ceef` also has 12 mixed-domain arithmetic cases and 12 pipeline/ancestor
cases (240 numerical comparisons). These are scoped cases, not exhaustive proofs.
Inspection and saved archives retain member identities, equations and fit records.
Parent-specific explorer creation/editing and hosted Colab remain separate
unaccepted work at this checkpoint.

## Native variables and fitted trees

| Inventory ID | Current entry point | State and qualification |
| --- | --- | --- |
| A1.structure.07 | `NativeGraphRequest`, `NativeCausalRecipe` softmax emission | SCOPED; actual integer categories, one-hot descendants and typed saved tasks; not complete TabAbax prior reproduction. |
| A1.structure.09 | `NativeDomainPrior` symmetric Dirichlet frequency weights | PARTIAL; retained Gamma draws provide logit offsets, not guaranteed marginal category frequencies; asymmetric vectors remain pending. |
| K1.structure.16 | Native argmax emission | SCOPED; deterministic first-class tie policy; complete Kumo prior is not reproduced. |
| M1.tree.05 | `FittedTreePlan(family="gbdt", task="classification")` | PARTIAL; frozen synthetic fitted classifier exists; source-specific mixed fit populations/lifecycles remain open. |
| M1.tree.06 | `FittedTreePlan(family="gbdt", task="regression")` | PARTIAL; frozen synthetic fitted regressor exists; source-specific fitting laws remain open. |
| I1.function.07 | XGBoost multi-output fitted-noise layer | PENDING; sklearn GBDT is not a substitute. |
| I1.code.09 | XGBoost histogram multi-output regressor | PENDING; no backend implementation or comparison. |
| OP.function.02 | Fitted XGBoost structural regressor | PENDING; support-only fitting lifecycle also remains separate. |

Native contracts and limitations are documented in
[sampled native graphs](sampled-native-graphs.md). Public fitted-tree exports,
immutable traversal, independent synthetic fitting and remaining source
lifecycles are described in [fitted structural trees](issue-28-fitted-structural-trees.md).
Independent fitted review includes float32 boundaries, GBDT initializers, class
probability/majority distinctions, ancestral fits, raw replay and budget rejection.
Original nonfinite-score and contradictory-metadata rejections remain preserved;
repairs have separate independent probes. This does not establish fitted-tree
explorer acceptance, XGBoost equivalence or predictive quality.

## Explicit next gaps

`K1.structure.09` (product with an MLP child) and `K1.structure.10` (product with a
cluster child) are PENDING at `723ceef`. The historical I2 product selector
intentionally contains only GP/tree/discretization/linear/quadratic children;
it must not be silently redefined to claim these Kumo options. A separate named
source variant is under implementation. Standalone residual/dropout/skip/gated
networks and genuine causal intervention requirements also remain open in the
full inventory; parent reductions and task-control pairs do not close them.
