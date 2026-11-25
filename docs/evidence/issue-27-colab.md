# Colab and local generation setup

Open [the notebook](../../notebooks/generation.ipynb) in Colab. It installs an immutable
source revision, explicitly selects CPU/CUDA, mounts Drive, runs a small generation
request, prints a same-workload estimate, and exposes a separate larger-run cell.
An unavailable CUDA device fails; it does not silently select CPU.

The notebook and `scripts/generate.py` call the same guarded generation interface.
Task count, time, bytes and worker count are explicit. Saved groups are checksummed;
predictor inputs and hidden truth remain separate. Restart uses the same directory
and request; failed-task retry must be selected explicitly. Use a single coordinator
per Drive directory. No distributed Drive locking or durable-commit claim is made.

Local reproduction, with the existing r0 environment selected:

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py python scripts/generate.py \
  --config configs/generation/smoke-cpu.toml --out /tmp/r0-27-platform-smoke --workers 2
# Repeat the identical command to verify completed tasks are skipped.
python3 scripts/run_check.py pytest tests/generation/test_guarded_job.py
```

The report regression first failed because `generated` was absent. It now checks
actual child execution and a restart with zero new generations. Peak RSS covers the
worker lifetime, including imports. CUDA peak allocation covers PyTorch allocations,
not total device use. Generation time includes serialization and persistence but
excludes interpreter startup. The outer elapsed time includes startup. Counts of
successful computations must not be confused with counts of committed tasks.

Local notebook generation/estimate cells were exercised with a temporary output
directory; install, Drive mount, browser interaction and download require Colab.
Colab and GPU acceptance are **NOT_RUN**. The embedded explorer and additional
families remain separate #27 PRs. The small linear smoke is not scale validation.

Colab setup follows the official [I/O notebook](https://colab.research.google.com/notebooks/io.ipynb)
and [runtime FAQ](https://research.google.com/colaboratory/faq.html).
