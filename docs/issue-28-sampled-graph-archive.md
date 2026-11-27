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
