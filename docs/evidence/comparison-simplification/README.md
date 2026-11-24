# Comparison simplification after PR 68

This refactor keeps the #26 recipes and fixture-only admission unchanged.
The shared `Predictor.fit(context)` / `predict(queries)` interface now serves
built-ins and external workers. `prediction_worker` owns validation, fit-once
invariance checks and the output record. `attempt` owns both execution paths;
even unenforced diagnostic attempts run in subprocesses.

## External configuration migration

`external_attempt`, `predict_adapter` and `adapter_worker` were removed. Use
`attempt`, `predict` and `prediction_worker`, respectively. External callers pass
`ExternalPredictor(name, python, config, worker=None, artifacts=())` instead of an
arbitrary command dictionary. CLI descriptors now use:

```json
{
  "name": "learner",
  "python": "/absolute/path/to/provisioned/python",
  "worker": "backend.py",
  "config": "backend-config.json",
  "artifacts": ["checkpoint.bin"]
}
```

Relative paths resolve beside the descriptor. The runner automatically hashes the
interpreter, worker, config and explicitly declared artifacts into cache identity.
A worker accepts `--config`, `--input`, `--out`; its config contains `identity`
(implementation, environment, settings, checkpoint, evidence), plus backend-specific
settings. Omit `worker` only for the built-in TabPFN worker. Changed files require
fresh attempts. Old descriptors must be migrated; old attempt bundles are not
reusable under the new source identity. Worker exit code 3 means unsupported.

## Evidence

- `check.log`: 202 passing tests, formatting, lint, types, module boundaries and caps.
- `parity-*.json`: all 45 retained pre/post-refactor fixture attempts have exactly
  equal predictions and scores, with byte-identical input/scoring files.
  Learner and TabPFN roles are authored mean/frequency substitutes, not models.
- `resource-probes.json` / `probes.log`: actual cgroup membership, timeout,
  memory limit, descendant termination, plus 45 successful enforced attempts.
- `smoke.log` / `eval-smoke.log`: exit 2; train and eval CLI commands are absent.
  No training loss or evaluation table is available.
- Independent standards and spec reviews found no actionable findings.
- An initial probe command used the wrong service name and was rejected. A second
  walkthrough rejected saved identity after source changed during execution.
  The retained successful run used stable source; neither failure was overwritten.

Reproduce using a provisioned environment, without installation or downloads:

```sh
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
make check
python3 scripts/run_check.py python scripts/comparison_external_walkthrough.py \
  --out /tmp/comparison-after-fresh
python3 scripts/run_check.py python docs/evidence/comparison-simplification/check_parity.py \
  /path/to/retained-before /tmp/comparison-after-fresh /tmp/parity
```

Generate `retained-before` by running the same walkthrough at base commit
`6c7242c31b96a294d58764d630bddc256b258ae3` in a separate checkout. For enforced
verification, use `scripts/comparison_resource_probe.py --out RECEIPT
--walkthrough-out FRESH_DIR` in a dedicated systemd user service named
`r0-comparison-probes-*`, with `Delegate=yes`, checkout working directory and
`PYTHONPATH=CHECKOUT/src`. See [PR 68 reproduction](../issue-26/completion.md).

## Real TabPFN smoke handoff — NOT_RUN

Modal is the intended platform; AGENTS.md and ADR 0004 require the guarded lane.
Live inspection on 2026-11-24 found `COMPUTE_ENABLED=false`, no provisioned wheel,
weights or compatible environment at the receipt's `/workspace/r0-assets/issue-25`
location, and no `scripts/modal_train.py` or `configs/runs/v0-cpu.toml` for the
factory workflow. This PR does not enable compute or supply that infrastructure.

The next run should use the three authored synthetic fixtures, pinned 2.2.1 wheel
and weight hashes, a compatible isolated environment, CPU float32, one CPU,
8 GiB and 300 seconds per attempt. Run actual classifier and regressor predictions,
retain raw outputs/runtime/fit-once invariance evidence, then score them. No real
benchmark data or training is needed. Before dispatch, the lane owner must provide
a comparison-capable guarded entrypoint and provisioned assets; the human owner
must set the spending cap and enable compute. Fixture gates still need an admitted
real-TabPFN execution route. Real compatibility and predictive quality remain
unverified until that run succeeds.
