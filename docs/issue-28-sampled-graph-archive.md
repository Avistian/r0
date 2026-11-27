# Issue 28: sampled graph archive and shared delivery candidate

This unaccepted candidate extends graph construction at `d56da97`. It does not
complete issue #28. Local UI/Colab delivery and remaining generator options,
categorical variables and interventions remain required.

`recipe_from_dict` recognizes `sampled-functional-graph-v1`. The more specific
`sampled_graph_from_dict(data, limits=...)` accepts caller-owned resource limits.
It bounds raw JSON before reconstructing request records or generator states,
rejects declared caps exceeding caller authority, and regenerates the complete
graph from the retained request and initial four states. Every retained field
must match, including raw roles and values, fits, owners, final states and runtime.
Unknown or missing fields are rejected. Comparison traverses the frozen graph
and admitted JSON directly, avoiding full dictionary/text copies of the graph.
This is replay-based validation, not proof of an external scientific provenance.

`generate_task` accepts the sampled graph, validates it, evaluates its frozen
runtime and stores the full construction in the recipe record. Cards identify
sampled frozen functions and retained four-stream construction provenance;
they do not claim authored provenance. `explain_generation` reconstructs the
same runtime for selected original rows, including repeated/reordered rows.
The decoder's strict record/array/pipeline helpers are shared with authored
functional decoding, whose behavior must remain compatible.

Sampled graphs expose their task controls. Changing task controls through the
pair interface regenerates from the same initial construction states and checks
the original graph first. Observation-noise pairs are exercised. Geometry
changes still require valid readouts; this does not implement arbitrary
structural interventions or add observed channels implicitly.

## Evidence status

Initial shared-decoder RED: `/tmp/r0-28-graph-archive-red.log`, unsupported sampled
recipe version. First integration passed save/reopen, generation, provenance and
inspection of rows `[7,0,7]`. Expanded coverage passed all 32 combinations of
eight function families and four root bases, including raw-role tampering,
missing final states and caller parameter limits:
`/tmp/r0-28-graph-archive-expanded.log` (32 passed, 44.69 seconds).

Further compatibility checks include regression/binary/multiclass tasks,
authored pipelines, structural noise and a coupled observation-noise pair.
The first run retained in `/tmp/r0-28-graph-archive-compatibility.log` had
100 passes and two invalid classification fixtures: observation noise is not a
classification control. Those fixtures now use separation; all three task-control
cases passed in `/tmp/r0-28-graph-archive-task-controls.log` (1.48 seconds).
Production behavior was unchanged by this fixture correction. Static gates
passed in `/tmp/r0-28-graph-archive-static-fifth.log`; earlier failures remain
retained. The final combined suite has not yet run.
Full-suite and independent review of this archive candidate are pending.
These paths are local author evidence, not a published portable archive.

Reproduce with the existing environment, without installing dependencies:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -ra tests/priors/test_sampled_graph_archive.py

UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

Memory admission remains conservative estimation rather than measured RSS.
GPU, browser, Colab and scientific evidence are not supplied by these tests.

## Source identity finding and repair candidate

Independent spec review rejected `6579ade`: changing only `context_rows` and
`split` through `graph.with_task()` left mechanisms, raw draws and all emitted
values unchanged, but changed `source_task_id`. The full archive duplicates task
controls under `request.task` and `runtime.task`; the shared completion function
previously excluded split controls only from a top-level task. The original
counterexample is retained at
`/tmp/r0-28-graph-archive-spec-6579ade/source_identity_probe.py` and `.log`.

The repair supplies a separate source fingerprint configuration excluding only
`context_rows` and `split` at both nested task locations. The replay archive keeps
the original fields. Source identity therefore distinguishes independently
sampled sources from additional episodes of the same source; episode identity
still includes partition controls. The regression first failed at
`/tmp/r0-28-graph-identity-red.log` (one case, 34 deselected). Repaired checks passed all 35 cases in 63.09 seconds, retained in
`/tmp/r0-28-graph-identity-green.log`. A separate three-case check
confirms genuine observation-noise or classification-separation changes still
change source identity: `/tmp/r0-28-graph-identity-controls.log` (3 passed,
37 deselected, 1.60 seconds). The deselected pair tests were not executed by
that filtered invocation. Independent review of this repair remains pending;
the historical REJECT is retained.

The earlier publication repair at `d56da97` completed `make check` with 1,705
passes and one skip in 532.92 seconds:
`/tmp/r0-28-graph-publication-check.log`. This verifies that earlier source,
not the subsequent archive integration or identity repair.

## Caller-owned exploration limits

Continued spec review rejected `d3f545b` because shared reopening and inspection
could not receive caller-owned limits, despite the specific sampled decoder
supporting them. A tiny graph declaring `parameters=1_000_001` generated and
reopened through the specific decoder, but failed through both shared entry
points. The retained original probe and output are
`/tmp/r0-28-graph-archive-spec-d3f545b/exploration_limits_probe.py` and `.log`.
The earlier source-identity repair passed independent review before this
separate finding; both historical rejections remain retained.

The repair threads explicit limits through the shared functional-graph paths:

```python
limits = priors.CausalLimits(parameters=1_000_001)
recipe = priors.recipe_from_dict(saved_recipe, limits=limits)
trace = priors.explain_generation(task, [7, 0, 7], limits=limits)
pair = priors.generate_pair(
    recipe, intervention="noise", value=0.25, seed=19, task_id=7, limits=limits
)
comparison = priors.compare_pair(pair.before, pair.after, intervention="noise", limits=limits)
```

The caller must supply these limits; saved payloads cannot grant themselves
larger budgets. Omitting the argument retains default caps. This parameter
controls frozen functional/sampled graph decoding, rather than replacing the
separate resource contracts of unrelated generator families.

Public regression RED: `/tmp/r0-28-graph-limits-red.log` (one failure because
`recipe_from_dict` lacked the argument). The repaired reopening/inspection test
passed in `/tmp/r0-28-graph-limits-green.log` (one pass, 35 deselected). Expanded
archive/pair/authored functional/generation compatibility checks passed all
60 cases in 61.24 seconds, retained in `/tmp/r0-28-graph-limits-compatibility.log`. Static gates passed in
`/tmp/r0-28-graph-limits-static-second.log`. No independent repair acceptance or
final full-suite claim is made by this note.
