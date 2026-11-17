"""scripts/check_compute_caps.py <config.toml>: exit 1 if this run would break a cap (ADR 0004)."""

import csv
import datetime as dt
import os
import pathlib
import sys
import tomllib

GPU_ORDER = ["none", "T4", "L4", "A10", "A100", "H100"]  # Verify against Modal's GPU docs
cfg = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
max_runs = int(os.environ.get("MAX_TRAIN_RUNS_PER_DAY", "2"))
max_gpu = os.environ.get("MAX_GPU", "none")
today = dt.datetime.now(dt.UTC).date().isoformat()
ledger = pathlib.Path("results/budget-ledger.csv")
runs_today = 0
if ledger.exists():
    with ledger.open() as f:
        runs_today = sum(1 for r in csv.DictReader(f) if r["timestamp_utc"].startswith(today))
gpu = cfg.get("compute", {}).get("gpu", "none")
problems = []
if runs_today >= max_runs:
    problems.append(f"{runs_today} runs today; cap is {max_runs}")
if GPU_ORDER.index(gpu) > GPU_ORDER.index(max_gpu):
    problems.append(f"config asks for {gpu}; MAX_GPU is {max_gpu}")
for key in ("max_steps", "max_wall_minutes"):
    if key not in cfg.get("training", {}):
        problems.append(f"training.{key} is missing")
if problems:
    print("Compute caps refuse this run:\n  " + "\n  ".join(problems))
    sys.exit(1)
print(f"caps ok: {runs_today}/{max_runs} runs today, gpu={gpu} (max {max_gpu})")
