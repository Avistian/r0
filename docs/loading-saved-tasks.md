# Load generated datasets without generating again

`load_group(path)` verifies a generation archive's completion receipt, checksum,
members and CRC, then yields its saved `TrainingTask` objects in manifest order.
It restores original row order from context/query indices. Each task contains the
same inputs, targets, scores, probabilities, identities and recipe/mechanism records
as the original generation. Arrays load on CPU; the execution record continues to
describe the hardware on which generation happened.

```python
from pathlib import Path
from r0.modules.generation import load_group
from r0.modules.explorer import inspect_task

archive = next(Path("run/groups").rglob("*.zip"))
for task in load_group(archive):
    episode = task.episode()  # shared training contract, regardless of generator family
    summary = inspect_task(task)
    print(episode.episode_id, summary["total_rows"])
```

Use the actual archive paths listed in your run's output. Loading allocates arrays
for one task at a time, independent of the number of tasks in the group. A caller
that collects the iterator into a list explicitly chooses to retain them all.
The caller can take bounded batches from the iterator or inspect one task at a
time. The loader invokes no generator and does not alter saved files.

This interface is for locally produced generation archives with their matching
receipts. It is not an importer for arbitrary uploaded ZIP files. Receipt hashes
detect changed output; they do not establish authenticity of untrusted files.
Keep diagnostic truth separate from predictor inputs when feeding a model.

Verification: `python3 scripts/run_check.py pytest tests/generation/test_loading.py -q`
round-trips regression, binary and multiclass tasks, compares complete episodes
and all diagnostic tensors, and rejects a changed archive before yielding a task.
