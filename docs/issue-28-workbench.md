# Sampled causal mechanism workbench

The local explorer and embedded Colab adapter share sampled causal construction,
inspection, editing, linked comparisons, save/reopen and export controls. This is
an implementation slice for #28, not full issue acceptance.

Open **Sample a causal graph** for an explicit DAG or Cauchy topology. Construction
controls include task geometry, node widths, observed/target channels, root
families, function/matrix priors, calibration rows and shared sampling policies.
The construction seed expands into four explicit NumPy streams; the sampled
archive retains their full states and all construction records. The dataset
seed/task identifier controls emitted rows separately. Every construction runs
inside the guarded worker, including archive replay and generation, subject to
the selected timeout. A failed or timed-out construction leaves the active task
unchanged and retains attempt records.

**Exploration caps** accepts explicit `CausalLimits` fields, including parameters,
node counts/channels and working bytes. Empty means defaults. Larger caps must
also be supplied when reopening the mechanism for execution, inspection or paired
comparison. They are transported through the worker request; the archive does
not grant itself larger authority.

**Inspect graph and prepare edit** opens a causal-structure diagram distinct from
the generation-process diagram. Node selection highlights its ancestors and shows
observed, target and hidden channel counts. Editable controls cover parent edges,
frozen function parameters, structural noise and observed/target assignments.
Dimension-changing edits must supply compatible function dimensions; invalid
edits fail visibly. The complete authored runtime remains editable as JSON.
Equations, original-row input/noise/output matrices and parameters are available
in the existing generation-process drill-down below the workbench.

Saving an edit creates an authored graph linked to the original sampled task.
Both archives remain saved. The comparison uses common plot axes, reports signal
spread and target changes, can be reopened, and exports the card plus both complete
replay bundles. Its kind is explicitly **mechanism-edit**. Shared seed is recorded;
exogenous-draw preservation under structural edits is not asserted. These are not
hard/soft causal interventions or observational conditioning. For causal graphs,
ordinary unchanged-mechanism task pairs expose only regression noise or
classification separation; feature-count/sparsity controls are not offered as
verified graph pairs.

## Verification and scope

Author checks are retained under ignored `docs/evidence/issue-28-workbench/`.
The local Chromium walkthrough is reproducible with:

```sh
python3 scripts/run_check.py python scripts/explore.py --out OUTPUT --port 18876
# Start an installed Chromium with a dedicated profile and CDP port 19228.
node scripts/workbench_browser_smoke.mjs 19228 http://127.0.0.1:18876/ EVIDENCE
```

The browser script verifies actual Cauchy construction, the causal diagram/node
editor, saved edit/reopen, a complete ZIP download, repeated-row reconstruction,
visible failure with preserved original data, and mobile overflow. It retains
screenshots and `receipt.json`. Python tests additionally verify explicit larger
caps through the worker, absence of parent-process construction, timeout receipts,
legacy replay reuse, and Colab callback generation/edit/reopen using substituted
external notebook APIs. This is local adapter verification, not hosted Colab proof.

Full-suite verification is parent-owned. Hosted Colab, GPU, native categorical
converters, hard/soft interventions, network-unit overlays, across-seed/filtering
dashboards and remaining source-option controls are still required. The generation
notebook's install revision must be updated to the integrated published commit
before hosted verification; the old pinned installation cannot provide these new
controls. No cloud or paid execution was performed here.

## Sampled geometry

The **Geometry-sampled causal graph** starting point also samples latent widths,
eligible feature-node subsets and channel assignments before mechanisms. Its
versioned archive retains that geometry and its named/raw decisions alongside the
mechanism construction. Controls expose node count, Cauchy offset, subset mode,
width-prior bounds, root/function priors, calibration and scalar structural noise.
The editor reports realized latent widths, converter-channel counts and assignments.
Readouts currently use identity numerical converters; classification targets are
adapter scores, not native categorical graph variables.

The geometry-specific browser flow uses the same script with a final argument:

```sh
node scripts/workbench_browser_smoke.mjs 19228 http://127.0.0.1:18876/ EVIDENCE \
  'Geometry-sampled causal graph'
```

Its local screenshots and receipt are in
`docs/evidence/issue-28-workbench/geometry-browser/`. Three focused backend cases
cover topology and geometry construction, larger caller caps, edit/reopen, and
worker timeout preservation. Hosted Colab proof remains separate and unexecuted.
