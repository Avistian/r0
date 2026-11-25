# One runner for linear, periodic and Fourier tasks

`generate_task(recipe, seed=..., task_id=..., device=...)` is the shared public
generation entrypoint. Existing family-specific functions remain available.
`recipe_from_dict` decodes versioned JSON/TOML records without changing the input.
Unknown versions fail; a request never silently substitutes linear generation.

`linear-v1` keeps its existing flat fields. `fourier-v1` contains family controls
and a nested `task` table for shared sizes, noise and classification controls.
The runner, subprocess worker and Modal adapter all use this decoder/entrypoint.
Saved archive cards contain the complete original recipe. Existing run directories
still require an identical request on resume.

The notebook now selects linear, periodic or RFF generation and uses separate
family/device output and measurement paths. It installs the pinned implementation
revision. Prior notebook output remains in its original directories; no migration
or overwrite is performed. Hosted Colab execution remains **NOT_RUN**.

Modal admission builds three 64-row diagnostic recipes from the selected family,
preserving mechanism parameters and dimensionality, while resetting noise and
class offsets/separation to neutral values. These diagnostic choices are validated
before submission. The requested training recipe stays unchanged. Older linear-only
hosted receipts do not verify hosted Fourier agreement or persistence.

## Reproduce

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/generation/test_family_runs.py tests/generation/test_launch_caps.py
python3 scripts/run_check.py python scripts/generate.py \
  --config configs/generation/fourier-cpu.toml --out /tmp/r0-fourier-run --workers 2
# Repeat exactly to verify saved tasks are skipped.
python3 scripts/run_check.py python scripts/generate.py \
  --config configs/generation/fourier-cpu.toml --out /tmp/r0-fourier-run --workers 1
```

Use `periodic-cpu.toml` with a new output directory for the other family.
The new end-to-end tests first failed on missing public dispatch. They now run
real child processes, read the persisted card, reconstruct its mechanism through
the public entrypoint, and resume with a changed worker count and no new tasks.
Admission tests verify that diagnostics retain the requested family and term count.
These are local checks; no paid work was launched for this slice.
