"""Same public-API fixture, run separately against base and repair source trees."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from r0.modules import priors

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
profiles = (
    "causal-affine-v1",
    "causal-activated-v1",
    "causal-preferences-v1",
    "causal-exogenous-v1",
    "causal-factorized-v1",
    "causal-importance-v1",
    "causal-matrices-v1",
    "causal-gaussian-matrices-v1",
    "causal-shared-matrices-v1",
    "causal-refreshed-matrices-v1",
    "causal-local-matrices-v1",
    "causal-global-matrices-v1",
)
controls = priors.LinearRecipe(
    rows=16, context_rows=8, features=2, signal_features=2, active_features=2
)
results = []
for profile in profiles:
    for family in ("forward", "layered", "redirection"):
        for seed in (28, 73):
            name = f"{profile}-{family}-{seed}"
            entry = {"name": name}
            try:
                recipe = priors.sample_recipe(
                    family, task=controls, seed=seed, task_id=0, profile=profile
                )
                raw = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
                (args.output / f"{name}.json").write_text(raw + "\n")
                entry["recipe"] = hashlib.sha256(raw.encode()).hexdigest()
                restored = priors.recipe_from_dict(json.loads(raw))
                assert restored == recipe
                task = priors.generate_task(restored, seed=seed, task_id=0)
                for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
                    tensor = getattr(task, field)
                    value = tensor.numpy().tobytes()
                    (args.output / f"{name}-{field}.bin").write_bytes(value)
                    entry[field] = hashlib.sha256(value).hexdigest()
                for field in ("source_task_id", "episode_id", "recipe_json", "mechanism_json"):
                    value = getattr(task, field)
                    entry[field] = hashlib.sha256(value.encode()).hexdigest()
                trace = priors.explain_generation(task, [0, 15])
                raw_trace = json.dumps(trace, sort_keys=True, separators=(",", ":"))
                (args.output / f"{name}-trace.json").write_text(raw_trace + "\n")
                entry["trace"] = hashlib.sha256(raw_trace.encode()).hexdigest()
                entry["outcome"] = "generated-replayed-traced"
            except ValueError as error:
                entry["outcome"] = "declared-failure"
                entry["error"] = str(error)
            results.append(entry)
            (args.output / "compatibility.json").write_text(json.dumps(results, indent=2) + "\n")
            print(name, entry["outcome"], flush=True)
imports = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("r0.") and path and path.endswith(".py"):
        imports[name] = {
            "path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
(args.output / "imports.json").write_text(json.dumps(imports, indent=2) + "\n")
