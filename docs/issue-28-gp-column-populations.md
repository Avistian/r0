# Finite GP feature-column populations

This component implements `OP.function.08`–`.11` at the public generation seam.
[O'Prior §2.1](https://arxiv.org/html/2605.18971v1#S2.SS1) describes RBF GP
feature columns via Cholesky and linear, quadratic or MLP targets. Locations,
kernel parameters, column independence and target parameter distributions below
are explicit r0 reconstructions, not disclosed historical O'Prior settings.
This is a component checkpoint, not whole-issue or peer-prior acceptance.

## Public construction

`GPColumnRequest` declares finite rows, columns, latent location dimensions,
shared or per-column `GPColumnKernel`, target family/output width, independent
calibration count and `CausalLimits`. `realize_gp_columns` produces an immutable
`GPColumnPopulation`: original row IDs, latent locations, innovations, complete
covariance and Cholesky factors, feature values, frozen target, independent
calibration inputs/scores/constants, raw/normalized targets and identity.
`GPColumnRecipe(request_or_population, LinearRecipe(...))` works through
`generate_task`, `recipe_from_dict`, `save_group`/`load_group` and
`explain_generation`. Stored task recipes contain the realized population.

Uniform or Gaussian latent locations, a one-dimensional grid, and explicit fixed
locations are named extensions. Kernels are
`amplitude**2 * exp(-squared_distance / (2 * length_scale**2))`. A shared kernel
shares its factor, not its normal column innovations. Separate kernels retain a
factor per column. Cholesky failure, nonfinite covariance geometry or output is a
reported failure: no jitter, clipping, eigendecomposition substitute or retry.

Targets are `x @ W + b`, `x @ W + b + x.T @ Q[k] @ x` per output, or
`tanh(x @ A + a) @ W + b`. Linear/neural matrices use independent standard-normal
entries scaled by inverse square root of input width; full quadratic matrices
use inverse input width. Biases are standard normal. These parameters remain
frozen across rows. All columns are declared structural inputs; the adapter's
active/signal feature counts must equal the GP-column count. Regression, binary
and multiclass adapters remain the shared implementations.

Target normalization uses independent draws from the single-row GP marginals,
with the same per-column amplitudes, evaluated through the frozen target.
Those calibration inputs are independent of all task locations, column draws,
context/query splits and labels. Constant calibration outputs retain scale one;
constants and raw calibration values remain inspectable. This is the named r0
independent-calibration adaptation, not support- or query-fitted normalization.

## Identity, comparisons and inspection

Population replay binds the complete request, seed, task ID and every retained
field, with strict numeric types and versioned schema. A realized population
cannot be generated under a different seed/task ID. Changing row count through
`with_task` returns a new construction request; a fixed-location request must
also supply consistent locations. Repeated/subset trace row selections index
the existing population without redrawing or subsetting its factorization.

`explain_generation` checks retained task metadata and tensors against replay,
then displays locations, innovations, GP feature values, raw targets, independent
normalization and shared target-adapter stages. `shared_matrices` preserves the
full covariance/factor with explicit original-row axes on both dimensions;
these matrices are not mislabeled as selected task-row values.

`compare_gp_columns` is a **target-mechanism edit**, not a general DAG do-operator.
Matched mode retains exactly the original GP population. Fresh mode requires an
explicit seed and draws new column innovations at the same locations/kernels.
Both modes preserve the original target; they evaluate it and the replacement
on the same selected population. Replacement calibration uses the retained
independent calibration inputs, never task/query rows. The comparison records
both score arrays and its full original/replacement provenance, and
`gp_column_comparison_from_dict` verifies matched/fresh replay. Classification
label resampling and full DAG root/channel interventions remain separate work.

## Admission and bounded verification

Geometry is admitted before RNG or numerical arrays. The dense reference guard
is 1,024 rows, 256 columns and 32 latent dimensions; caller parameter/memory caps
can admit substantially less. Kernel covariance storage is quadratic in rows;
Cholesky work is cubic. These are implementation guards, not measured supported
maxima or claims to the general 20,000-row envelope. Aggregate comparison
admission precedes original-population replay and includes replacement target
work. Estimates include retained/replay copies; they are not process RSS.

Public tests retain RED/GREEN evidence for literal Cholesky, all target/adaptor
combinations, archive/repeated-row inspection, matched/fresh comparisons,
nonfinite/singular rejection and comparison admission ordering. Negative probes
also cover forged identity/values and rejection before RNG/array allocation.
The last admission RED exposed original replay before aggregate rejection;
that history remains in the evidence directory. Adjacent generation traces,
recursive populations and prototype-batch tests were included in focused checks.

Outside the test suite, the retained script computes a two-row RBF covariance
with off-diagonal one-half, factor entries one, one-half and sqrt(3)/2, and feature
values two and one. Literal targets give 4 (linear), 11 (quadratic), and
-0.5231883119115297 (neural). Independent scalar-loop arithmetic also reconstructs
sampled targets and calibration for all three families. Raw packets and outputs
are retained locally under `docs/evidence/issue-28-gp-columns/`; they are gitignored.
Full-suite, smoke, browser, Colab, CUDA and scale measurements were not run in
this worker. Author numerical checks do not replace independent frozen-head
verification.

## Required follow-on work

Shared hybrid DAG/root integration remains REQUIRED: feed retained finite column
values with exact row IDs through an explicit external-population context.
Do not expose the population as an iid root law or ordinary frozen scalar GP
function. Whole-workbench controls, editing/comparison save-reopen flows,
local-browser and Colab demonstrations, resource measurements and independent
verification remain REQUIRED. This core adds no UI presets and supplies no
acceptance for those paths. Public archive/replay and numerical inspection are
not browser evidence.

Feature selection and production filtering remain #30. Row-axis convolution,
VAR and temporal mechanisms remain #39; exchangeable finite GP row covariance
does not establish temporal-model coverage. No downstream utility or full-prior
reproduction claim is made.

## Outer-envelope preflight repair

Independent review of `db5140b` retained **REJECT**: malformed task rows,
comparison mode or replacement width caused four RNG initializations before
rejection. Its separate 24-case numerical check passed; numerical agreement did
not remove that admission defect. The original review packet remains unchanged
in the rejected worktree's ignored `docs/evidence/issue-28-gp-independent/`.

The repair validates explicit outer schema/version, task controls and adapter
geometry, comparison mode/fresh identity and replacement dimensions before
population replay. Both direct comparisons and comparison decoding use the same
control validator. Pure metadata checks also validate original population identity,
row IDs, target dimensions, draw/kernel schema and retained matrix/vector shapes.
A request-only recipe validates adapter compatibility before replacing its source
with a replayed population. Numerical constructors and sampled laws are unchanged.

The expanded public RED exposed 16 failures among 20 malformed-envelope cases;
all 20 now reject with zero RNG initializations. The original outside-test probe,
changed only to expect zero invalid-case calls and write into the repair directory,
passes. The independent numerical script was rerun unchanged by the repair author:
24 cases pass. Eighteen valid artifact groups spanning three target families,
three task types, shared/per-column kernels and both matched/fresh comparisons
are byte-identical before and after repair (canonical JSON SHA-256
`ea8a6f2af14e89851f96c420bdcb67d8df201725571d555316b5740de9105a9c`).
Fifty focused tests pass. Evidence is retained under ignored
`docs/evidence/issue-28-gp-envelope-preflight/`. Fresh independent repair review is
still required; no full suite, browser, Colab or GPU run was performed here.
Hybrid integration and whole-workbench delivery remain required follow-on work.
