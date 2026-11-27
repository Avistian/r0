import argparse
import copy
import hashlib
import json
import pathlib
import sys
import traceback
from dataclasses import asdict

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession

ap = argparse.ArgumentParser()
ap.add_argument("outputdir")
a = ap.parse_args()
out = pathlib.Path(a.outputdir)
out.mkdir(parents=True, exist_ok=True)


def save(n, x):
    (out / n).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda x: x.tolist()))


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
results = []
controls = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
base = p.sample_causal_recipe(
    p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
    task=controls,
    seed=28,
    task_id=0,
    widths=(2, 1, 3, 1),
)
rec = p.with_source_activations(
    base,
    prior=p.SourceActivationPrior(mode="meta", placement="both", calibration_rows=32),
    seed=28,
    task_id=0,
)
original = json.loads(json.dumps(asdict(rec)))
save("original.json", original)


def reject(name, change):
    d = copy.deepcopy(original)
    change(d)
    save(name + "-input.json", d)
    try:
        p.recipe_from_dict(d)
    except ValueError as exc:
        results.append({"name": name, "status": "REJECTED", "error": str(exc)})
        save("results.json", results)
    else:
        raise AssertionError("CORRUPTION ACCEPTED: " + name)


def at(path, value):
    def change(d):
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value

    return change


try:
    for key in ("input_mean", "input_inverse_std", "shift", "output_mean", "output_inverse_std"):
        reject(
            "alter-" + key,
            at(
                ("nodes", 0, "output_activation", key, 0),
                original["nodes"][0]["output_activation"][key][0] * 0.75 + 0.125,
            ),
        )
    for key, value in [
        ("calibration_digest", "0" * 64),
        ("calibration_id", "wrong-stream"),
        ("shift_rows", [0, 0]),
        ("calibration_rows", 31),
    ]:
        reject("alter-" + key, at(("nodes", 0, "output_activation", key), value))
    reject("binding-missing", lambda d: d["source_bindings"].pop())
    reject("binding-duplicate", at(("source_bindings", 1), original["source_bindings"][0]))
    reject("binding-rescale", at(("source_bindings", 0, "rescale_draw"), 1))
    reject("primitive", at(("nodes", 0, "output_activation", "kind"), "identity"))
    reject("prior-policy", at(("source_prior", "policy"), "refresh"))
    reject("prior-mode", at(("source_prior", "mode"), "global"))
    reject("prior-selector", at(("source_prior", "selector"), "plain"))
    reject("seed", at(("calibration_seed",), 29))
    reject("task-id", at(("calibration_task_id",), 1))
    reject(
        "noise-order",
        at(
            ("nodes", 0, "noise_position"),
            "after_output"
            if original["nodes"][0]["noise_position"] == "before_output"
            else "before_output",
        ),
    )
    reject(
        "inactive-root-input",
        at(("nodes", 0, "input_activation"), {"kind": "tanh", "parameters": []}),
    )
    reject(
        "extra-draw",
        lambda d: d["rescale_preferences"]["draws"].append(d["rescale_preferences"]["draws"][0]),
    )
    reject("rr-scale-name", at(("rescale_preferences", "draws", 0, "request", "name"), "different"))
    task = p.generate_task(rec, seed=28, task_id=0)
    card = json.loads(task.mechanism_json)["resource_admission"]

    # Independent integer bookkeeping for retained parameter arrays and buffer formula.
    def request_cost(q):
        return 2 + len(q["shape"])

    def numerical(q):
        return sum(
            request_cost(x["request"]) + len(x["units"]) + (2 if x["mode"] == "meta" else 0)
            for x in q["preferences"]
        ) + sum(
            request_cost(x["request"]) + 1 + len(x["units"]) + len(x["values"]) for x in q["draws"]
        )

    # Existing preference records supply their validated counts.
    # New integration cost is independently enumerated.
    metadata = (
        4
        + len(rec.noise_units)
        + rec.source_selections.parameter_count
        + numerical(original["rescale_preferences"])
        + sum(2 + (b.rescale_draw is not None) for b in rec.source_bindings)
    )
    expected = (
        sum(len(n.bias) * (2 + sum(len(rec.nodes[j].bias) for j in n.parents)) for n in rec.nodes)
        + metadata
    )
    for n in rec.nodes:
        for act in (n.input_activation, n.output_activation):
            expected += len(act.parameters) + (
                6 * len(act.input_mean) + 2 if isinstance(act, p.CalibratedActivation) else 0
            )
    widths = [len(n.bias) for n in rec.nodes]
    fan = max(sum(widths[j] for j in n.parents) for n in rec.nodes)
    extra = max(
        [
            10 * len(act.input_mean)
            for n in rec.nodes
            for act in (n.input_activation, n.output_activation)
            if isinstance(act, p.CalibratedActivation)
        ]
    )
    working = (
        8
        * max(rec.task.rows, rec.source_prior.calibration_rows)
        * (6 * sum(widths) + 2 * fan + 2 * controls.features + 4 * len(rec.target) + 4 + extra)
        + 128 * expected
    )
    assert card["parameters"] == expected
    assert card["estimated_working_bytes"] == working
    reject("parameter-limit", at(("limits", "parameters"), expected - 1))
    reject("working-limit", at(("limits", "working_bytes"), working - 1))
    admitted = copy.deepcopy(original)
    admitted["limits"]["parameters"] = expected
    admitted["limits"]["working_bytes"] = working
    assert p.recipe_from_dict(admitted).limits.parameters == expected
    save(
        "admission.json",
        {
            "actual": card,
            "independent_parameters": expected,
            "independent_working_bytes": working,
            "exact_boundary": "PASS",
        },
    )
    for mode in ("local", "global", "meta"):
        for policy in ("shared", "refresh"):
            prior = p.SourceActivationPrior(
                selector="plain", mode=mode, policy=policy, placement="both"
            )
            plain = p.with_source_activations(base, prior=prior, seed=28, task_id=0)
            save("plain-" + mode + "-" + policy + ".json", asdict(plain))
            assert not plain.rescale_preferences.draws
            assert all(
                type(n.input_activation) is p.CausalActivation
                and type(n.output_activation) is p.CausalActivation
                for n in plain.nodes
            )
            assert p.recipe_from_dict(json.loads(json.dumps(asdict(plain)))) == plain
            assert plain.source_selections.categorical.policy == policy
            results.append({"name": "plain-" + mode + "-" + policy, "status": "PASS"})
    for selector in ("standard", "plain"):
        for typ, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
            name = selector + "-" + typ
            session = ExplorerSession(out / (name + "-original"))
            request = {
                "record": {
                    "recipe": {"rows": 16, "context_rows": 8, "task_type": typ, "classes": classes},
                    "seed": 28,
                    "task_id": 0,
                },
                "sampling": {"family": "layered", "profile": f"causal-source-{selector}-global-v1"},
            }
            save(name + "-request.json", request)
            generated = session.request("generate", request)
            save(name + "-generated.json", generated)
            trace = session.request("graph", {"indices": [0, 15]})
            save(name + "-trace.json", trace)
            assert trace["graph"]["target_max_error"] < 1e-10
            reopened = ExplorerSession(out / (name + "-original")).request(
                "open", {"archive": generated["archive"], "task_id": 0}
            )
            assert reopened["view"] == generated["view"]
            replay = ExplorerSession(out / (name + "-replay"))
            regenerated = replay.request("generate", {"record_json": generated["recipe_json"]})
            assert regenerated["recipe_json"] == generated["recipe_json"]
            assert replay.request("row", {"index": 15}) == session.request("row", {"index": 15})
            results.append({"name": name, "status": "PASS"})
    save("results.json", results)
    save("summary.json", {"status": "PASS", "checks": len(results)})
except Exception:
    save("results.json", results)
    save("summary.json", {"status": "STOPPED", "traceback": traceback.format_exc()})
    raise
