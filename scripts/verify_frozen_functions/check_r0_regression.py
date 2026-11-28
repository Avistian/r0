"""Actual BASE/SOURCE public recipe, task generation and save/load comparison."""

import hashlib
import itertools
import json
import platform
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import r0
from r0.modules import priors as p
from r0.modules.generation import load_group, save_group, verify_group

ROOT = Path(__file__).resolve().parent
revision = sys.argv[1]
assert revision in ("base", "source")
assert Path(r0.__file__).resolve().is_relative_to(ROOT / revision)
OUT = ROOT / f"r0-{revision}"
OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
profiles = ("causal-affine-v1", "causal-source-standard-v1", "causal-source-standard-matrices-v1")


def encode(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.tolist(),
    )


results = []
for profile, family, task_type in itertools.product(
    profiles, ("forward", "layered"), ("regression", "binary", "multiclass")
):
    task_id = len(results)
    name = f"{profile}-{family}-{task_type}"
    case = {
        "name": name,
        "profile": profile,
        "family": family,
        "task_type": task_type,
        "seed": 649,
        "task_id": task_id,
    }
    try:
        controls = p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        )
        recipe = p.sample_recipe(family, profile=profile, seed=649, task_id=task_id, task=controls)
        recipe_json = encode(asdict(recipe))
        decoded = p.recipe_from_dict(json.loads(recipe_json))
        assert decoded == recipe
        task = p.generate_task(decoded, seed=649, task_id=task_id)
        task_json = encode(asdict(task))
        archive = OUT / f"{name}.zip"
        receipt = save_group(archive, [(task_id, task)], max_bytes=4_000_000)
        assert verify_group(archive) == receipt
        loaded = list(load_group(archive))
        assert len(loaded) == 1
        assert encode(asdict(loaded[0])) == task_json
        with zipfile.ZipFile(archive) as saved:
            member_hashes = {
                member: hashlib.sha256(saved.read(member)).hexdigest()
                for member in sorted(saved.namelist())
            }
        case.update(
            status="completed",
            recipe=json.loads(recipe_json),
            generated_task=json.loads(task_json),
            trace=p.explain_generation(task, [15, 0, 8]),
            archive_members_sha256=member_hashes,
        )
    except (ValueError, ArithmeticError) as exc:
        case.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    payload = encode(case)
    (OUT / f"{name}.json").write_text(payload + "\n")
    if revision == "source":
        expected = (ROOT / "r0-base" / f"{name}.json").read_text()
        assert expected == payload + "\n", f"BASE/SOURCE byte difference: {name}"
    results.append(
        {
            "name": name,
            "status": case["status"],
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    )
    print(f"{case['status']} {name}", flush=True)
imports = {}
for name, module in sorted(sys.modules.items()):
    if name == "r0" or name.startswith("r0."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(ROOT / revision)
        imports[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
receipt = {
    "revision": revision,
    "python": sys.version,
    "executable": sys.executable,
    "numpy": np.__version__,
    "torch": torch.__version__,
    "platform": platform.platform(),
    "threads": torch.get_num_threads(),
    "cases": results,
    "imports": imports,
    "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "completed": sum(case["status"] == "completed" for case in results),
    "failed": sum(case["status"] == "failed" for case in results),
    "byte_equal_cases": len(results) if revision == "source" else None,
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(encode({k: receipt[k] for k in ("revision", "completed", "failed", "byte_equal_cases")}))
