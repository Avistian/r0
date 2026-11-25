# Sample settings for each saved task

Add a `[sampling]` table to a generation TOML request to choose new mechanism
settings for each task identity. `[recipe]` supplies `LinearRecipe` task controls:
shape, task type, active features, noise and classification difficulty. The caller
chooses one family and one versioned [sampling profile](sampling-profiles.md).
This does not select production mixture weights or filter difficult tasks.

```toml
[recipe]
rows = 128
context_rows = 96
features = 8
signal_features = 8
active_features = 4

[sampling]
family = "polynomial"
profile = "broad-v1"

[run]
seed = 27
count = 8
group_size = 4
max_seconds = 60
max_bytes = 10485760
device = "cpu"
```

Run the checked-in example locally, then run the identical command again to resume:

```bash
python3 scripts/run_check.py python scripts/generate.py \
  --config configs/generation/profile-polynomial-cpu.toml \
  --out /tmp/r0-sampled-polynomial --workers 2
```

Use `UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv` when deliberately
reusing an environment. In Colab, the same `load_request` and `run_guarded` entrypoints
accept this request and a Drive output directory. The guarded Modal adapter decodes
the same request; its admission plan selects the requested family for its separate
regression, binary and multiclass CPU/GPU checks. This implementation does not prove
that a particular profile has run successfully in either hosted environment.

## Saved requests and restart behavior

Sampled runs use `generation-run-v2`. Static recipe runs retain their existing
`generation-run-v1` headers and explorer directory identities. Do not change an
existing run's configuration: use a new output directory.

Before admitting a group, the coordinator selects every pending task's concrete
recipe and writes them to that attempt's `start-N.json`, keyed by task identifier.
The requested profile and original task controls stay in `run.json`. Proposal
bytes count against the storage limit; an oversized proposal stops before task
generation. When a persistence callback is configured, it runs before admission.
Hard runtime limits still belong to `run_guarded` or the hosted watchdog.

Completed task cards retain both the actual selected recipe and a `sampling`
object containing `family`, `profile` and original `task` controls. Numerical
failures retain their selected recipes in the group start record and their reasons
in the failure log. Resume verifies completed archives and skips completed or failed
tasks; only `--retry-failed` attempts failures again, with the same recipe and task
identity. Earlier attempts remain present. No automatic resize or family substitution
occurs.

## Explorer generation and replay

The shared local/Colab `ExplorerSession` accepts fresh sampling choices beside
the task record:

```python
from pathlib import Path
from r0.modules.explorer import ExplorerSession

session = ExplorerSession(Path("/tmp/r0-sampled-explorer"))
result = session.request(
    "generate",
    {
        "record": {
            "recipe": {"rows": 32, "context_rows": 20},
            "seed": 27,
            "task_id": 9,
        },
        "sampling": {"family": "polynomial", "profile": "broad-v1"},
    },
)
replayed = session.request("generate", {"record_json": result["recipe_json"]})
```

Replay restores the original controls and validates that they select the recorded
recipe. A conflicting recipe or sampling override is rejected. A replay in another
output directory generates the same data; a replay in the same directory resumes
the verified archive. The local and embedded Colab explorer expose these choices
under **Mechanism settings**. The notebook smoke cell exposes `PROFILE` and `FAMILY`.

## Reproduce the integration walkthrough

Commit source before running:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/sampled_run_walkthrough.py \
  --out docs/evidence/issue-27-sampled-runs
```

The walkthrough retains eight saved polynomial tasks, a separate worker's resume
report, direct replay from concrete settings, graph reconstruction, a fresh explorer
replay and an exported bundle. Its report identifies source and driver hash.
`docs/evidence/` is ignored; raw runs are not committed. This is a bounded CPU
integration check, not broad-profile resource qualification or a training result.
