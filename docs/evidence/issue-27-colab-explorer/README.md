# Embedded Colab explorer preparation

Hosted Colab acceptance: **NOT_RUN**. These are local checks, not browser or Drive
evidence. The shared adapter source is pinned to
`8794ad9581eafa3fa17ef77d279613058512acdc` in the notebook.

`make check` passed: 439 passed, one CUDA skip. `make smoke` and `make eval-smoke`
each exited 2 because the repository CLI has no `train` or `eval` command yet;
there is no training-loss summary or evaluation table. An offline wheel build
succeeded and its archive contained the notebook adapter and all three shared
web assets. The external Colab/IPython transport test uses stubs and does not
claim hosted execution.

The retained `linear.json` and `composition.json` come from executing the actual
notebook generation, resume, estimate and archive-verification cells locally.
Both generated eight fresh tasks in two groups and resumed with zero generated
and eight completed. The second execution starts with a fresh Python namespace
before the estimate cell has ever run, checking that timing already survived.
The helper skips package installation, Drive mounting and the embedded UI. The
larger-run switch stays false. Reproduce from this checkout with fresh outputs:

```bash
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/colab_notebook_walkthrough.py \
  --out /tmp/r0-colab-restart-linear

UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/colab_notebook_walkthrough.py \
  --out /tmp/r0-colab-restart-composition --family Composition
```

This is an adapter and notebook preparation slice of #27. Generation graphs,
response curves and controlled pairs remain separate implementation work. Hosted
installation, actual JavaScript interaction, downloads, GPU operation and durable
Drive restart must still be checked in Colab using the notebook checklist.
