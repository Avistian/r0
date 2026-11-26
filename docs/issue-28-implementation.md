# Issue #28 implementation and evidence ledger

The [accepted specification](specs/0028-causal-network-generators.md) is the
authoritative scope. This ledger tracks execution, not a smaller replacement goal.

## Checkout and scope preservation

- Working branch: `feat/issue-28-generators`, worktree `/tmp/r0-28`.
- Fetched base: `13f49167460058f847d8a651af489e578783a6c0`.
- The older home checkout and its unrelated edits remain intact.
- The accepted design was copied before edits; its fingerprint is retained in
  local `docs/evidence/issue-28/preflight.txt`. Only issue #28 glossary additions
  were appended to current main's vocabulary.
- `make preflight tracked-files` passed using the existing home-repo virtualenv
  with `scripts/run_check.py`; it resolved `r0` to this worktree's `src`.
- The oversized interview document was separated into the authoritative scope and
  its supporting research appendix to comply with the repository's file-size rule.

## Baseline verification

The public generation/composition, saved task-group and explorer-session baseline
passed **19 tests** before implementation. Command:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_compositions.py \
  tests/generation/test_task_groups.py tests/explorer/test_session.py
```

This verifies the selected inherited behavior and environment, not the new #28
requirements or full repository acceptance.

## Public verification seams

These are the public interfaces covered by the user's accepted generator/replay,
intervention, inspection and delivery requirements (decisions 2, 8, 10, 14, 18):

| Seam | What it proves | What it does not prove |
| --- | --- | --- |
| `priors.generate_task` and recipe decoding | Recipe behavior, typed episodes, randomness separation and reproducibility | Browser behavior or predictive superiority |
| Public recorded-mechanism inspection and intervention operations | Worked structural arithmetic, node/channel roles and intervention invariants | Inferring causal models from observed data |
| Generation run/group save/load | Identity, failures, bounded output, verified restart and durable replay | Hosted execution without a target-environment run |
| Explorer session and actual browser/Colab transports | User controls, comparisons, diagrams, exports and reopen behavior | UI proof from API tests alone |

Use one failing public-behavior test followed by its implementation per slice.
Expected results come from authored calculations or independent invariants, not
the implementation under test. Full completion still requires all specification
requirements and option records to have appropriate evidence.

## Execution stages

| Stage | State | Completion evidence |
| --- | --- | --- |
| Primary-source option extraction and finite inventory | READY | Source/option record for every documented relevant choice, including variants and scope boundaries |
| Replayable explicit causal and neural mechanisms | IN_PROGRESS | Independent worked calculations through shared public interfaces |
| Topology, function, activation, weights, roots/noise and hyperprior sampling | IN_PROGRESS | Every inventory option exercised with traceable recipe and evidence |
| Mixed typed variables and native/adapted targets | NOT_STARTED | Schema, probabilities, held-out-label separation and saved roundtrips |
| Matched/population interventions and graph/mechanism edits | NOT_STARTED | Independent invariants, versioned identities and saved comparisons |
| Diagnostics and coverage/filtering evidence | IN_PROGRESS | Original/retained denominators, failure reasons and structural/functional diversity |
| Shared explorer/Colab workbench and exports | NOT_STARTED | Actual interactions and rendered artifacts in each environment |
| Generation/storage/parallel/restart integration | NOT_STARTED | Retained execution, failure/restart and resource measurements |
| Independent review, checks and publication | IN_PROGRESS | Fixed-revision review, required repository checks, PR and approved-specification link |

## Important semantic constraints

- An arbitrary observed subset of a causal model does not define a deterministic
  function from its observed columns to its target. Mechanism replay/intervention
  must not masquerade as conditional prediction or feed observed descendants back
  into an invented forward function. The explorer must label the requested operation.
- Constant, saturated and disconnected but numerically valid draws remain valid.
  Calibration must represent zero dispersion explicitly rather than silently
  filtering these draws or dividing by zero.
- Mechanisms and fixed calibration are sampled independently of query rows and
  labels. Inspection must not change the task. New edits have new identities.
- Original generation failures remain visible; no silent retries, substitutions,
  clipping, implicit CPU fallback or claims that a narrow representative covers a
  complete published family.
- Source-specific neural samplers and trees fitted to synthetic random data are
  distinct constructions. Standalone tree breadth remains #29, but all documented
  structural component variants remain required #28 scope.

## Outstanding authorization and evidence

This implementation goal does not itself authorize a paid/full-scale launch,
merge, deployment, or real-data download. Local code and synthetic verification
can proceed. Hosted measurements remain pending until executed under the existing
authorized path. Draft [PR #125](https://github.com/Avistian/r0/pull/125) publishes
the specification. Issue #28 links the verified scope at `66e91e2`; fresh readback
and the saved before/after bodies confirm the original issue body is unchanged.
Reproduce using `python3 scripts/verify_issue28_spec_link.py` (read-only `gh api`
access). Its [receipt](research/0028-spec-link-receipt.json) retains the exact body
snapshots and observed published-file hash. The procedure compares published bytes
with the pinned Git object, verifies the 19 numbered decisions and final paper-option
condition, checks the sole appended section, then reads back the current issue.

## Source gate result

[The finite inventory](research/0028-source-option-inventory.md) contains 581
source/extension/disposition records and 140 specification anchors. The eight JSON
chunks have unique IDs and explicit ownership/unknown states. These are traceability
records, not unique algorithms or implementation counts. Exact peer/checkpoint
reproduction is not claimed. Documented choices are ready for implementation;
unknown unpublished algorithms remain explicit.

## Local hardware

The inspected environment reports PyTorch `2.15.0+cpu`, 12 CPU threads, no CUDA
runtime and no available CUDA device. GPU execution evidence is NOT_RUN here.
Historical sandboxed Pyright logs contain dependency-resolution failures. The
exact source states and complete reproduction context for those exploratory runs
are NOT_RETAINED, so they are not exact-revision verification evidence. Subsequent
commit hooks ran typechecking at the identified commits. Local exploratory logs
remain under `docs/evidence/issue-28/`; a current typecheck is reproducible with the
repository wrapper and the explicitly selected interpreter/environment.

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
