# Population and converter workflow checkpoint

The combined source is a65fbb9f5180fbd2b570399880d7c9f6b2369d8e.
Full make check passed: 2,428 tests, one skip, one warning in 1,250.47 seconds;
all static gates passed. The warning is the retained sklearn class-count fixture.
Exact-head logs and receipt remain ignored under
docs/evidence/issue-28-population-converter-release in the release worktree.
This is a bounded implementation checkpoint, not whole-issue or predictive acceptance.

## Explicit fitted populations and sampled parameters

Support-only fitting uses actual context rows, including one-row contexts.
Batch-reference fitting uses the actual whole episode; it is deliberately
distinct from the query-independent support-only contract. Mixed-independent
populations retain numerical/categorical construction and source-qualified
Mitra-inspired adaptations. Fits are frozen once, archived and replayed without
implicit refitting. Parameter laws compose with these lifecycle choices; an
unused row-count law rejects rather than pretending it controls support size.

Independent lifecycle review covers four backends, two fitted stages,
support/query independence with a ball root and noise, literal mixed-population
arithmetic, custody mutations, no-sklearn replay, and local Chromium workflows.
Independent sampled-population review adds forty scalar-law/engine cases,
120 invalid metadata/law cases, multi-stage support invariance and four-engine
rollback after an actual failed backend fit. Integrated public checks and local
browser verification preserve large task IDs, frozen Next, repeated-row traces
and the saved baseline after invalid input.

These are explicit sklearn-backed adaptations. XGBoost remains unimplemented;
neither GBDT naming nor a shared parameter law establishes backend equivalence.
See [lifecycle contracts](issue-28-fitted-population-lifecycles.md) and
[sampled population parameters](issue-28-sampled-population-fits.md).

## Typed converter workflows

The seven modes now connect to local construction, inspection, authored edits,
saved comparisons and hard, soft and population interventions. Capture-only
surgery preserves descendant state; representation surgery changes descendant
inputs explicitly. Original-row traces distinguish the equation, stochastic
emission, captured category and represented state.

Original C candidate a67e8f was rejected for omitting the original uniform parent
when capture-only surgery preserved a stochastic softmax representation. Repair
487c6df restores that dependency. Independent continuation covers 31 public
checks: all seven modes, partial-channel NumPy arithmetic, soft replacement,
population identity, overwritten-equation bypass, metadata/custody mutations and
admission. Actual Chromium verification covers exact IDs, capture intervention,
the repaired trace edge, repeated rows, export/reopen and invalid-action state
preservation. Original rejection evidence remains retained.

Source-specific fit lifecycles and complete paper reproduction remain separately
qualified. Local callback support is not hosted Colab evidence. See
[converter contracts](issue-28-categorical-converters.md).

## Scope and evidence

The combined candidate passed 129 focused checks, static gates, commit hooks and
a Chromium family-selector integration check before the full run. Population
integration separately passed 128 focused cases and a local browser scenario.
Those counts are separate runs with overlapping cases, not additive coverage.

Network input views, independent variance matching and their editor/guard
repairs are later candidates outside this source. Neural hyperparameter profiles,
internal noise, remaining root/topology laws, all unimplemented inventory options,
hosted/GPU execution and scientific acceptance remain open. Generated evidence
is ignored under docs/evidence; source and reusable drivers remain tracked.
