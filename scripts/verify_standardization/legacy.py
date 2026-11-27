"""Fresh bounded legacy public-profile comparison capture."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors as p

REPO = Path(os.environ["REPO"]).resolve()
GIT_REPO = Path(os.environ["GIT_REPO"]).resolve()
OUT = Path(os.environ["OUT"]).resolve()
PIN = os.environ["PIN"]
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
profiles = [
    "causal-" + s + "-v1"
    for s in [
        "affine",
        "activated",
        "preferences",
        "exogenous",
        "factorized",
        "importance",
        "matrices",
        "gaussian-matrices",
    ]
]
profiles += ["causal-" + s + "-matrices-v1" for s in ["shared", "refreshed", "local", "global"]]
profiles += [
    "causal-source-" + style + mode + matrix + "-v1"
    for matrix in ["", "-matrices"]
    for style in ["standard", "plain"]
    for mode in ["", "-refreshed", "-local", "-global"]
]


def enc(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


records = []
for profile in profiles:
    for family in ["forward", "layered", "redirection"]:
        for task_id in [3, 17]:
            record = dict(profile=profile, family=family, seed=649, task_id=task_id)
            try:
                recipe = p.sample_recipe(
                    family,
                    profile=profile,
                    seed=649,
                    task_id=task_id,
                    task=p.LinearRecipe(rows=16, context_rows=9),
                )
                record["recipe"] = asdict(recipe)
                task = p.generate_task(recipe, seed=649, task_id=task_id)
                record["task"] = asdict(task)
                record["trace"] = p.explain_generation(task, [15, 0, 9])
                record["status"] = "PASS"
            except (ValueError, RuntimeError) as error:
                record.update(status=type(error).__name__, error=str(error))
            records.append(record)
            with (OUT / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(record, default=enc, sort_keys=True) + "\n")
(OUT / "records.json").write_text(json.dumps(records, default=enc, sort_keys=True) + "\n")
modules = {}
for name, module in tuple(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if not name.startswith("r0") or not file:
        continue
    file = Path(file).resolve()
    rel = file.relative_to(REPO)
    raw = file.read_bytes()
    pinned = subprocess.check_output(["git", "show", f"{PIN}:{rel}"], cwd=GIT_REPO)
    modules[name] = dict(
        path=str(file),
        sha256=hashlib.sha256(raw).hexdigest(),
        pinned_sha256=hashlib.sha256(pinned).hexdigest(),
        matches=raw == pinned,
    )
(OUT / "custody.json").write_text(
    json.dumps(
        dict(
            pin=PIN,
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            modules=modules,
        ),
        indent=2,
    )
    + "\n"
)
assert all(v["matches"] for v in modules.values())
(OUT / "summary.json").write_text(
    json.dumps(
        dict(
            total=len(records),
            completed=sum(r["status"] == "PASS" for r in records),
            explicit_errors=[
                {k: r[k] for k in ["profile", "family", "task_id", "status", "error"]}
                for r in records
                if r["status"] != "PASS"
            ],
            sha256=hashlib.sha256((OUT / "records.json").read_bytes()).hexdigest(),
        ),
        indent=2,
    )
    + "\n"
)
