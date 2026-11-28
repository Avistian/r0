import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import asdict

from r0.modules import priors as p

ap = argparse.ArgumentParser()
ap.add_argument("outputdir")
a = ap.parse_args()
out = pathlib.Path(a.outputdir)
out.mkdir(parents=True, exist_ok=True)


def save(n, x):
    (out / n).write_text(
        json.dumps(x, sort_keys=True, indent=2, allow_nan=False, default=lambda x: x.tolist())
    )


def sha(b):
    return hashlib.sha256(b).hexdigest()


save(
    "execution.json",
    {
        "script_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "imports": {
            str(pathlib.Path(m.__file__).resolve()): sha(pathlib.Path(m.__file__).read_bytes())
            for n, m in list(sys.modules.items())
            if n.startswith("r0.") and getattr(m, "__file__", None)
        },
    },
)
profiles = [
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
]
results = []
for i, profile in enumerate(profiles):
    fam = ["forward", "layered", "redirection"][i % 3]
    typ, classes = [("regression", 0), ("binary", 2), ("multiclass", 3)][i % 3]
    controls = p.LinearRecipe(rows=16, context_rows=8, task_type=typ, classes=classes)
    name = profile + "-" + fam + "-" + typ
    recipe = p.sample_recipe(fam, profile=profile, task=controls, seed=28, task_id=0)
    save(name + "-recipe.json", asdict(recipe))
    try:
        task = p.generate_task(recipe, seed=28, task_id=0)
        save(name + "-task.json", asdict(task))
        trace = p.explain_generation(task, [0, 15])
        save(name + "-trace.json", trace)
        results.append({"case": name, "status": "PASS", "trace_error": trace["target_max_error"]})
    except ValueError as e:
        results.append({"case": name, "status": "EXPLICIT_FAILURE", "error": str(e)})
save("results.json", results)
