# Bounded episode consumption: retained execution

Runtime and drivers: `268a60199c773f3af8ab6304db44ccb44e82dd01`.
Base: `27103c8f546220160475a6e91d8e93699671a1eb` (PR #110).
No library, production training, evaluation, notebook or UI code changes here.

| Source | Episodes | Optimizer steps | Last batch | Peak process RSS, bytes |
| --- | ---: | ---: | ---: | ---: |
| Generated, all ten families × three task types | 30 | 8 | 2 | 337,559,552 |
| Same sweep extended to 90 task identities | 90 | 23 | 2 | 337,522,688 |
| Saved group, generation disabled during consumption | 7 | 2 | 3 | 335,200,256 |

Both generated runs retain at most four episodes per batch, 256 total rows and
4,032 context/query distance cells across that batch. Independent verification
checks unique source identities, all predictions and losses, finite-difference
gradients, optimizer updates, shapes, batch boundaries and summaries. Across all
three runs, maximum absolute prediction difference is `4.440892098500626e-16`,
loss difference `1.7763568394002505e-15`, gradient difference
`2.2395738041858237e-11`, update difference `6.938893903907228e-18`.
The 30- and 90-task runs include one and three episodes respectively with classes
absent from context. The saved run includes two. None were filtered or relabeled.
The 90-task run reuses the first run's 30 identities and adds 60; identity uniqueness
is checked within each run, and the two generated runs contain 90 distinct tasks total.

These are single-run consumption observations, not generator throughput studies,
100M-task scaling proof, GPU verification or learned-model performance evidence.
Elapsed time starts before optimizer initialization and excludes imports/process
startup. RSS includes Torch/Python and allocator caches. See `environment.json`.

## Reproduce

From this checkout with the existing verification environment selected:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/run_check.py python scripts/consume_generated_batches.py --out /tmp/batch-new-30
python scripts/run_check.py python scripts/verify_generated_batches.py /tmp/batch-new-30 \
  --output /tmp/batch-new-30/verified.json
python scripts/run_check.py python scripts/consume_generated_batches.py \
  --count 90 --out /tmp/batch-new-90
python scripts/run_check.py python scripts/verify_generated_batches.py /tmp/batch-new-90 \
  --output /tmp/batch-new-90/verified.json
python scripts/run_check.py python scripts/saved_batch_walkthrough.py --out /tmp/batch-new-saved
python scripts/run_check.py python scripts/verify_generated_batches.py \
  /tmp/batch-new-saved/consumption --output /tmp/batch-new-saved/consumption/verified.json
python scripts/run_check.py python scripts/batch_failure_walkthrough.py \
  --saved /tmp/batch-new-saved/consumption --out /tmp/batch-new-failures
```

To inspect the exact retained raw output, extract `raw.tar.gz` into a new directory.
`files-*.json` records every uncompressed file's length and SHA-256. This preserves
complete per-query NPZ predictions, JSONL batch records, the actual saved groups
and their receipts, and all malformed/failure examples. The small JSON files beside
this README are copies for quick inspection. Use `--archive` when verifying a saved
run after extraction so the verifier reads the relocated `saved/source-group.zip`.
Write fresh verifier output outside the original run directory.

Recheck the retained package and extract it without changing its records:

```python
import hashlib
import json
import tarfile
from pathlib import Path

evidence = Path("docs/evidence/issue-27-batch-consumption")
environment = json.loads((evidence / "environment.json").read_text())
bundle = evidence / "raw.tar.gz"
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == environment["raw_tar_sha256"]
with tarfile.open(bundle) as archive:
    count = 0
    for manifest in evidence.glob("files-*.json"):
        for record in json.loads(manifest.read_text()):
            data = archive.extractfile(record["path"]).read()
            assert len(data) == record["bytes"]
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            count += 1
    assert count == environment["raw_files"] == 65
    archive.extractall("/tmp/batch-retained", filter="data")
```

Then verify the relocated saved output:

```sh
python scripts/run_check.py python scripts/verify_generated_batches.py \
  /tmp/batch-retained/saved/consumption --archive /tmp/batch-retained/saved/source-group.zip \
  --output /tmp/batch-retained/relocated-verified.json
```

The failure walkthrough rejects a saved 256-row task in the middle of a batch,
retains that task's exact identity and recipe, and confirms no batch update was
reported. It also deliberately edits otherwise valid saved receipts to stop after
four of seven tasks, or consume seven when only six were requested. Both are
rejected and produce no successful verification record.

## Review and checks

Initial source `461fce1` passed `make check`: **472 passed, 1 skipped**.
Two independent review axes then rejected the initial change: saved-task failures
lacked task identity/recipe (spec); saved-source verification did not check early
termination or excess consumption (standards). Commit `268a601` corrects both and
adds the executed failure walkthrough. Both reviewers accepted the source delta;
these historical findings are not erased by the final passing runs.

All runtime evidence here was freshly produced after that correction. Commit hooks
checked final script formatting/lint and repository guards. No `src/` or tests
changed after the full suite, so it was not repeated for the script-only fixes.
`r0 train` and `r0 eval` remain unavailable; the bounded example does not implement
those commands or claim their smoke checks pass. Hosted Colab and GPU acceptance
remain separate pending work for #27.
