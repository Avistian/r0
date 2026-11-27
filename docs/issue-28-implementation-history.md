# Issue 28: earlier implementation checkpoints

These are retained historical checkpoint entries, moved verbatim from the
[active implementation ledger](issue-28-implementation.md). Their scoped results
and then-current pending states are historical; the active ledger and accepted
specification govern current completion.

## First causal core slice

Implemented explicit affine scalar/vector DAGs through `recipe_from_dict` and
`generate_task`, with independently keyed structural innovations, original-row
trace reconstruction, partial-channel readouts and exact recipe replay. Fixed
independent calibration is optional; constant channels retain their raw value
with identity normalization and a diagnostic flag. Shared nonzero-edge channel
ancestry is labeled as potential dependence, not proof of predictive usefulness.

Five passing public-behavior cases and historical failing logs are retained under
`docs/evidence/issue-28/` (intermediate RED source snapshots are NOT_RETAINED):
`causal-authored-*`, `causal-trace-*`, `causal-vector-*`, `causal-calibration-*`,
and `causal-constant-*`. This is a foundation slice, not completed paper-family
coverage. Sampled topology/function priors, nonlinear mechanisms, typed categorical
states, interventions and the expanded workbench remain outstanding.

Independent review rejected the first core checkpoint for direct outcome exposure;
see [the review and reproduction record](issue-28-core-review.md). The first
regression fix has a committed RED state; the original review is retained.


The direct exposure defect is fixed at `fb5d4d1`, with committed RED `9406302`.
Resource admission has committed RED `1a8d889`; limits now explicitly bound graph
geometry, parameters and estimated working storage before generation. These are
admission estimates, not measured resource-qualification claims. See the review
record for defaults and reproducible commands; broad inventory execution remains
outstanding.


Scoped independent spec recheck **PASS** at `b507f0c`: direct target exposure,
all five resource limits, invalid-limit rejection, calibration admission,
structural arithmetic, RNG identity, selected-row traces and JSON replay passed.
This is a delta confirmation, not overall #28 acceptance. The original REJECT is
retained. The expanded focused integration command passed **33 tests** at the same
source revision; full-suite, browser, GPU and Colab acceptance remain pending.

## Sampled topology checkpoint

At `b03dc24`, forward Bernoulli DAGs, sparse layered DAGs and an explicitly
reconstructed predecessor-redirection tree sampler are available. Scalar/vector
recipes retain actual topology, mechanisms and disjoint readouts. The
`causal-affine-v1` profile routes these through the existing bounded save/reopen
and explorer-session paths. See [laws and evidence](issue-28-topology-sampling.md).

`make check` passed at that source revision: **497 passed, 1 skipped**, with all
earlier preflight/static gates passing. Local log:
`docs/evidence/issue-28/topology-make-check.log`. `make smoke` and `make eval-smoke`
remain NOT_AVAILABLE: the current CLI only defines replay/oracle/diagnose-priors;
T7/T12's train/eval commands have not landed. This is not hosted CI or ML acceptance.

Independent spec review: **PASS for the partial topology slice**, not full #28.
Independent standards review found one P2 evidence-retention gap in the publication
claim; the procedure and receipt above repair it. The original finding remains
recorded in the topology checkpoint document. Scoped standards recheck PASS at
`30eb1b1` independently confirmed the retention repair. No package source changed after the
reviewed revision. Broader graph families, nonlinear/tree/network mechanisms,
typed categorical states, interventions, workbench UI and final environment/scale
evidence remain required and incomplete.

## Nonlinear structural checkpoint

`causal-v2` adds explicit input/output activation placement and structural-noise
order, frozen numeric parameters, row-wise channel transforms and saturation
inspection. The 42 operation names include aliases and boundary conventions;
they are not 42 unique algorithms. The limited `causal-activated-v1` sampling
profile reaches generation, save/reopen and row inspection. See the
[definitions, source mapping and verification](issue-28-nonlinear-mechanisms.md).

At `fbb1aa8`, `make check` passed **567 tests, 1 skipped** and all static gates.
The earlier 562-pass suite at `28a478f` did not detect three independent numerical
findings: reciprocal-power intermediate overflow, lost integer-exponent parity,
and avoidable additive-log rejection. Their original REJECT remains retained;
committed RED `d6d368e` and repair `fbb1aa8` address them. Scoped independent review
and its results are recorded in the nonlinear checkpoint document.

An author CPU census retained all **192/192 valid** draws, including disconnected,
inactive, saturated and zero-dispersion cases. This verifies one small convenience
profile, not full paper-option coverage or throughput. Noise/root distributions,
weight hyperpriors, GP/Fourier/tree/neural constructions, wider graphs, native
typed variables, interventions, workbench UI and final environment/scale evidence
remain required. Smoke/eval-smoke remain NOT_AVAILABLE; browser, Colab and GPU
acceptance remain PENDING. The full #28 goal is still active.


## Activation preference checkpoint

The shared sampler now supports independent categorical, per-task Dirichlet and
homogeneous activation choices, with frozen uniform/log-uniform/integer numeric
parameter distributions. `SampledCausalRecipe` retains the prior, realized
probabilities and actual node parameters without changing legacy recipe fields.
The new `causal-preferences-v1` exploration profile reaches generation, saved
explorer sessions, row inspection and replay. [Laws, ranges and evidence](issue-28-activation-priors.md)
remain explicit about source-specific selectors and numerical correlations still
pending; this implements the activation part of decision 11, not the mechanism
and weight preference requirements.

Independent spec review originally rejected an approximately one-hot homogeneous
record that should have been rejected exactly. RED `3eafcd5` and GREEN `ec06de3`
retain that repair. Standards review also identified a missing evidence attribution
and an ambiguous internal profile selector; both are resolved with the historical
finding retained. `make check` at `ec06de3` passed **579 tests, 1 skipped**, and all
static gates; the complete earlier and repaired outputs are published in the
checkpoint record. An author census at `c92a551` retained **192/192 valid** tasks
across three modes, with flagged cases intact and no filtering. Full source
coverage, wider mechanisms/noise/weights, typed variables, interventions, full
workbench delivery and final environment/resource acceptance remain required.


## Exogenous and structural-noise checkpoint

Authored stochastic causal nodes now retain Gaussian, standardized uniform,
Student-t and symmetric Gaussian-mixture laws, optional conditional truncation,
uniform-volume ball geometry and frozen correlation factors. Explicit responses
scale structural noise by the pre-noise affine signal, separately from observation
noise. Raw exogenous draws, actual scales and innovations have distinct row traces.
Finite-moment certification gates optional fixed normalization while raw heavy-tail
tasks remain available. The `causal-exogenous-v1` exploration profile reaches
sampled scalar/vector tasks and saved explorer backend workflows. See [laws and
source boundaries](issue-28-exogenous-laws.md).

Independent review retained an avoidable intermediate-overflow REJECT at
`f75c9cd`, then a symmetric underflow REJECT at `684a3ec`. Committed RED cases and
repairs preserve both findings. At `a47b14e`, **135 independent checks pass**,
including 73 high-precision boundary cases and 18 exact legacy snapshots across
all three older profiles. Standards review finds no hard violations; one
nonblocking duplicated-registry heuristic remains. [Verification, retained
failures and executable procedures](issue-28-exogenous-verification.md).

Complete `make check` at `a47b14e` passes **597 tests, 1 skipped**, with all static
gates passing. An author census at `f75c9cd` retains **191 valid / 192 attempted**
tasks; the one square-overflow failure and every sampled recipe remain recorded,
with no retries or filtering. All four laws, both geometries and sampled response
strengths occur. This remains a small CPU convenience-profile census. Broader
paper-specific root/noise laws, weights, mechanisms/networks/trees, typed states,
interventions, workbench delivery and final environment/resource qualification
remain required. Smoke/eval-smoke are NOT_AVAILABLE; browser, Colab and GPU
acceptance remain PENDING. The full #28 goal is active.


## Factorized projection checkpoint

`FactorizedWeights` and `MatrixCausalNode` retain the left/right factors, inner
coefficients and realized dense projection through decoding, row inspection and
replay. Shape/provenance checks reject contradictory records; admission includes
the factor storage before construction. `causal-factorized-v1` samples Gaussian
factor products with explicit inner-dimension and scaling laws through all three
existing topology families and the saved explorer backend. [Definitions, source
boundaries and evidence](issue-28-factorized-weights.md) distinguish these factors
from singular-value decompositions and retain all remaining weight-family scope.

Independent review rejected avoidable intermediate overflow at `ec522ed`.
RED `209b4e6` and repair `ef1bacc` preserve finite extreme products using bounded
exact arithmetic when ordinary intermediates are unsafe. The resource-accounting
heuristic is resolved at `ec39b46`. At that revision, **33 independent check groups
pass**, including Fraction oracles, 72 successful sampled recipes, three saved
backend workflows and exact matching of **252 legacy outcomes** (250 successes,
two preserved failures). Complete `make check` passes **605 tests, 1 skipped**,
with all static gates passing. The author census retains **192/192 valid** tasks
and 1,739 factorized projections without filtering or retry.

This delivers a factor-product construction and an r0 exploration law. Source
positive-weight hyperpriors, importance/kernel/transformed matrices, masks,
normalization/jitter stages, shared weight preferences, wider mechanisms and
networks, typed variables, interventions, workbench UI and final environment/
resource qualification remain required. The full goal is not complete.


<!-- End of verbatim historical checkpoint entries. -->

## Importance-weighted projection checkpoint

At `e6d8faa`, authored uniform and polynomial/log-Gaussian importance vectors feed
recorded row-normalized Gaussian projections. `causal-importance-v1` exposes this
through all three topology families and shared save/reopen/replay. Stable centered
arithmetic preserves decay under huge common perturbations and valid epsilon-zero
directions after weight underflow. Family provenance rejects factorized/importance
substitutions. [Definitions, source distinctions and evidence](issue-28-importance-weights.md).

Repository checks pass **613 tests, 1 skipped**; source/test bytes match `fc0265b`
where that check began. Independent scoped review passes 2,134 numerical checks,
384 sampled recipes, four saved backend sessions and 315 exact old-profile outcomes
(including two failures). Resource instrumentation precedes all 22 matched Gaussian
allocations in its declared example. The author census retains **191 valid / 192
attempted**, including the failed integer-power recipe. A disconnected-graph oracle
mistake is retained as a harness error, not a product rejection. Standards found no
hard violations; repeated profile registries remain a nonblocking heuristic.

This supplies positive-vector/weighted-Gaussian options within W01/W03, not all
weight priors. Shared source numerical preferences, kernel/transformed matrices,
wrappers, masks, networks/trees, typed states, interventions and the full workbench
remain required. CPU backend evidence does not establish browser/Colab/GPU or
scientific acceptance. The full goal stays active.


## Matrix families and kernel point checkpoint

At `7eea0bd`, the constructor pipeline retains Gaussian, importance, positive-vector
factorized, signed Laplace and flattened-activation matrices; explicit jitter and
normalization stages; and covariance/Gaussian-mixture kernel point constructions.
`causal-matrices-v1` samples all five families, while `causal-gaussian-matrices-v1`
restricts the source family. Both share the existing generation, inspection and
saved replay workflow. [Definitions and source distinctions](issue-28-matrix-families.md)
retain the paper/code point variants and explicitly local sampling choices.

Full checks pass **623 tests, 1 skipped**, all static gates passing. Independent
review passes 224 numerical checks, 72 sampled recipes (71 successes, one retained
square-overflow failure), six saved backend workflows, and 378 exact legacy outcomes
(including two failures). Standards found no hard violations; duplicated component-
probability normalization is a nonblocking heuristic. The corrected zero-coefficient
oracle assumption remains a separate harness error, not a product rejection.

Unfiltered author censuses retain **192/192 valid** mixed-profile tasks and
**190/192 valid** Gaussian-only tasks. Both Gaussian-only failed recipes and the
independent failed recipe are retained completely, with reconstruction hashes.
Timing and peak RSS are NOT_MEASURED. Browser/Colab/GPU evidence remains outstanding.

This advances I2 matrix options and their primitive point laws. Shared source
numerical/categorical preferences, complete source activation weighting, wider
graph/network/tree/GP mechanisms, typed states, interventions and the full workbench
remain required. No whole-issue or complete source joint-prior acceptance is claimed.
Earlier implementation entries were moved verbatim to the linked checkpoint history
to keep files within the repository size limit. The full goal remains active.
