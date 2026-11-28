# Root option audit

This audit checks 23 root-related inventory records against implementation
`784d8af`: P1.root.01–03, P1.code.02–06, P2.root.01–04 and I2.root.01–11.
The table records implementation paths and qualifications, not whole-paper
reproduction or an independent reviewer verdict. P1.root.03 remains an unresolved
source disclosure; it is not silently counted as implemented by another law.

For I2, the pinned
[`_points.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py)
was checked again. `RandomPoints` selects uniform, ball, Gaussian or covariance
base points and then applies a width-preserving random function. Its separate
Gaussian-mixture class supplies centers, masses, component factors and scales;
it is not a fifth choice in that default selector. The implementation preserves
this distinction through joint roots and joint point clouds respectively.

## Option-level map

All named entry points are available through `r0.modules.priors`.

| ID | Implementation / disposition | Qualification |
| --- | --- | --- |
| P1.root.01 | `SourceRootCoordinate("gaussian")`, `draw_source_root`, `SourceRootBase` | Explicit mean/scale or retained sampled profile; zero scale remains a valid constant before optional calibration. |
| P1.root.02 | `SourceRootCoordinate("zipf", exponent=...)` | Unbounded support is distinct from the clipped code branch. Float output can round integers above the exact binary64 range; raw int64 draws are retained. |
| P1.root.03 | **UNDISCLOSED source algorithm** | The paper's multivariate label does not specify an algorithm. Correlated, mixture and prototype constructions provide named alternatives, not proof of the undisclosed source law. |
| P1.code.02 | `SourceRootCoordinate("unit_uniform")`; `SourceRootProfile` uniform selector | Unit-uniform values differ from symmetric-uniform values. Unused constructor Gaussian parameters remain in sampled-profile custody. |
| P1.code.03 | `sample_source_root_profile` with `presample_gaussian=True` | Frozen Gaussian parameter vectors are bound to model/selection snapshots and raw draws. The retained lifecycle is explicitly named. |
| P1.code.04 | Standard Gaussian coordinate and normal profile | No calibration is silently added. |
| P1.code.05 | Categorical coordinate/profile with retained probabilities; `sample_source_root_base` | Frozen independent normalization is a named adaptation. The separately named batch-reference emitter normalizes its own batch. Numeric root codes alone do not declare native categorical observations. |
| P1.code.06 | Clipped Zipf profile with retained sampled exponent and clip at ten | Unclipped integers remain in the raw ledger. Independent centering and generated-batch centering are distinct lifecycles. |
| P2.root.01 | Gaussian source-coordinate or joint root | Source scale hyperpriors are not inferred from the availability of a configurable Gaussian law. |
| P2.root.02 | Symmetric-uniform source-coordinate or joint root | The P2 selector's fixed unit half-width is a documented reconstruction. |
| P2.root.03 | `SourceRootProfile(selector="p2-node-mixture-r0")` | One choice per logical root vector, with explicit normal probability; not independent coordinate choices or a claimed complete source hyperprior. |
| P2.root.04 | `sample_prototype_root`, `draw_prototype_root`, `PrototypeRootBase`; separate batch-reference API | Shared row weights form convex mixtures of a frozen bank. Count rounding and multinomial temperature rules are explicit reconstructions; probability-temperature and Dirichlet modes are extensions. |
| I2.root.01 | `JointRootPrior(families=("uniform",))` and uniform `JointRootBase` | Symmetric hypercube base, followed by the selected function when using a joint root plan. |
| I2.root.02 | Ball joint root | Gaussian direction and independent radius `U**(1/width)` produce volume-uniform base points; not a uniform-radius sphere. |
| I2.root.03 | Gaussian joint root | Base draws and transformed output remain separately inspectable. |
| I2.root.04 | Covariance joint root with owned `JointWeightPlan` | Raw Gaussian factor columns are weighted before applying the factor; no matrix-normalization substitution. |
| I2.root.05 | `sample_joint_root`, `materialize_joint_root`, `sample_joint_root_values` | Mandatory width-preserving function plan; frozen calibration is retained and fresh emissions do not refit. |
| I2.root.06 | Covariance root with uniform base choice | Base selector is recorded after the complete factor-weight subtree. |
| I2.root.07 | Covariance root with Gaussian base choice | Same owned factor, distinct base law; snapshots and selected law survive decoding. |
| I2.root.08 | `sample_joint_points`, `materialize_joint_points` | Gaussian center arrays and component assignments belong to the separate Gaussian-mixture point cloud. |
| I2.root.09 | Joint point mixture-weight plan and realized masses | Assignment probabilities use normalized realized masses; zero total mass rejects without replacement. |
| I2.root.10 | Joint point Gaussian component matrices with factor-column weights | A single cloud is shared across a kernel matrix batch; records retain factor geometry and draws. |
| I2.root.11 | Joint point shared scale mean/coefficient and component draws | Component factors use the retained exponential scales. These are not prototype convex-mixture temperatures. |

## Verification and saved arithmetic

The inspected implementation includes `source_root_profile_sampling.py`,
`sampled_source_root_base.py`, `joint_root_plans.py`, root realization/emission,
and the joint point/matrix and prototype records. Existing tests cover:

- Explicit six-coordinate equations and raw replay across five RNG engines;
  source selector parameter binding and once-per-root P2 choices.
- Independent frozen calibration, constant rejection, query-prefix stability,
  exact metadata tampering, and retained sampled-root calibration provenance.
- Prototype count conventions, normalized-count/Dirichlet arithmetic, shared
  coordinate dependence, batch-versus-frozen lifecycle and graph archives.
- All four joint base families crossed with eight function families; fresh
  emission without fitting, independent base arithmetic and decoder rejection.
- Joint mixture point ownership, component draw order and shared-cloud kernel
  arithmetic in `test_joint_point_plans.py` and `test_joint_matrix_plans.py`.

Fresh focused logs are retained under gitignored
`docs/evidence/issue-28-root-option-audit/`. A separate coordinate calculation
stores 128 rows for Gaussian, unit uniform, symmetric uniform, categorical,
clipped Zipf and unbounded Zipf outputs, plus an independently drawn NumPy
reference. All six output columns agree exactly in this calculation. It preserves
the raw Zipf draw (maximum 88) and the explicit clip at ten (output maximum 10).
The script and arrays are retained beside the result JSON. This bounded CPU
arithmetic is not a distribution-quality or downstream-usefulness measurement.

The contracts in [source roots](issue-28-source-root-laws.md),
[prototype roots](issue-28-prototype-roots.md),
[prototype graph integration](issue-28-prototype-root-graphs.md),
[joint roots](issue-28-joint-root-realization.md) and
[joint matrices/points](issue-28-joint-matrices.md) retain the detailed limits.
Some of their earlier sections describe historical prerequisites; later sections
and the current implementation supersede those historical pending statements.

## Remaining work

P1.root.03 still lacks a disclosed source algorithm. Complete source hyperprior
and selection reproduction, broad joint resource/diversity qualification and
independent review are not established by this audit. Current full checks remain
separate. All other #28 inventory records remain active; this table neither
removes them nor moves generator work to #126.
