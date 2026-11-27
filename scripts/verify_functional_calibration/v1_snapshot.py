import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve() / "src"))
import hashlib
import json

from r0.modules import priors as p

out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
manifest = {}
for family in ["uniform", "gaussian", "ball", "covariance"]:
    root = (
        p.JointRootBase(family, 1)
        if family != "covariance"
        else p.JointRootBase(
            family,
            1,
            gaussian_factor=((2.0,),),
            weights=p.UniformWeightVector((0.5,)),
            factor=((1.0,),),
            covariance_base="gaussian",
        )
    )
    for kind, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
        name = f"{family}-{kind}"
        controls = p.LinearRecipe(
            rows=19,
            context_rows=7,
            features=1,
            signal_features=1,
            active_features=1,
            task_type=kind,
            classes=classes,
        )
        node = p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (2.0,), (3.0,), (-1.0,))),
            root=root,
            noise_scale=(0.1, 0.2, 0.3, 0.4),
        )
        recipe = p.FunctionalCausalRecipe(
            controls,
            (node,),
            ((0, 0),),
            tuple((0, i + 1) for i in range(3 if kind == "multiclass" else 1)),
        )
        task = p.generate_task(recipe, seed=119, task_id=23)
        values = {}
        for key in [
            "inputs",
            "scores",
            "targets",
            "probabilities",
            "context_indices",
            "query_indices",
        ]:
            value = getattr(task, key)
            if value is not None:
                a = value.numpy()
                data = a.tobytes()
                (out / f"{name}-{key}.bin").write_bytes(data)
                values[key] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "dtype": str(a.dtype),
                    "shape": a.shape,
                }
        for key in ["recipe_json", "mechanism_json", "source_task_id", "episode_id"]:
            value = getattr(task, key).encode()
            (out / f"{name}-{key}.txt").write_bytes(value)
            values[key] = hashlib.sha256(value).hexdigest()
        graph = json.dumps(
            p.explain_generation(task, [18, 0, 18]), sort_keys=True, separators=(",", ":")
        ).encode()
        (out / f"{name}-graph.json").write_bytes(graph)
        values["graph"] = hashlib.sha256(graph).hexdigest()
        manifest[name] = values
(out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
print("Wrote 12 v1 task snapshots using", p.__file__)
