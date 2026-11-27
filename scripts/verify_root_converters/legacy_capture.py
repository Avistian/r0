import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from r0.modules import priors as p

torch.set_num_threads(1)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
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
    "causal-source-" + s + mode + matrix + "-v1"
    for matrix in ["", "-matrices"]
    for s in ["standard", "plain"]
    for mode in ["", "-refreshed", "-local", "-global"]
]
rows = []


def enc(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    raise TypeError(type(x).__name__)


for profile in profiles:
    for family in ["forward", "layered", "redirection"]:
        for taskid in [3, 17]:
            row = dict(profile=profile, family=family, seed=649, task_id=taskid)
            try:
                recipe = p.sample_recipe(
                    family,
                    profile=profile,
                    seed=649,
                    task_id=taskid,
                    task=p.LinearRecipe(rows=16, context_rows=9),
                )
                row["recipe"] = asdict(recipe)
                task = p.generate_task(recipe, seed=649, task_id=taskid)
                row["task"] = asdict(task)
                row["trace"] = p.explain_generation(task, [15, 0, 9])
                row["status"] = "PASS"
            except (ValueError, RuntimeError) as e:
                row.update(status=type(e).__name__, error=str(e))
            rows.append(row)
            with (out / "attempts.jsonl").open("a") as log:
                log.write(json.dumps(row, default=enc, sort_keys=True) + "\n")
(out / "records.json").write_text(json.dumps(rows, default=enc, sort_keys=True) + "\n")
modules = {
    name: {
        "path": module.__file__,
        "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    }
    for name, module in tuple(sys.modules.items())
    if name.startswith("r0")
    and getattr(module, "__file__", None)
    and Path(module.__file__).is_file()
}
(out / "custody.json").write_text(
    json.dumps(
        dict(
            python=platform.python_version(),
            numpy=__import__("numpy").__version__,
            torch=torch.__version__,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            imports=modules,
        ),
        indent=2,
    )
    + "\n"
)
(out / "summary.json").write_text(
    json.dumps(
        dict(
            proposals=len(rows),
            passed=sum(r["status"] == "PASS" for r in rows),
            failed=sum(r["status"] != "PASS" for r in rows),
            sha256=hashlib.sha256((out / "records.json").read_bytes()).hexdigest(),
        ),
        indent=2,
    )
    + "\n"
)
