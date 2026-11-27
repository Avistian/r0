"""Independent public-API checks; no test helpers or private implementation calls."""

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession

OUT = Path(__file__).resolve().parent
SEED, TASK = 119, 23


def save(name, value):
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def rng(prefix, *keys):
    wire = json.dumps(
        ["functional-causal-v1", SEED, TASK, [*prefix, *keys]],
        sort_keys=True,
        separators=(",", ":"),
    )
    return np.random.default_rng(int(hashlib.sha256(wire.encode()).hexdigest(), 16))


def population(rows, prefix=()):
    x = -1 + 2 * rng(prefix, "root-base", 0).random((rows, 1))[:, 0]
    state0 = 3 * (x + 0.2 * rng(prefix, "innovation", 0, 0).normal(size=rows))
    state1 = 3 * (2 * x + 0.4 * rng(prefix, "innovation", 0, 1).normal(size=rows))
    nonlinear = state0**2 + 0.7 * state1 + 3
    downstream = nonlinear + 0.15 * (0.5 + 0.2 * np.abs(nonlinear)) * rng(
        prefix, "innovation", 1, 0
    ).normal(size=rows)
    return np.column_stack([2 * x, downstream])


controls = p.LinearRecipe(rows=19, context_rows=7, features=1, signal_features=1, active_features=1)
nodes = (
    p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0,), (2.0,))),
        root=p.JointRootBase("uniform", 1),
        noise_scale=(0.2, 0.4),
        pipeline=p.CausalNodePipeline(
            2, converters=(p.NumericalConverter(1),), node_importance=3.0
        ),
    ),
    p.FunctionalCausalNode(
        (0,),
        p.FrozenQuadraticFunction(
            2, (0, 1), (((1.0, 0.0, 0.0), (0.0, 0.0, 0.7), (0.0, 0.0, 3.0)),)
        ),
        noise_scale=(0.15,),
        noise_response=p.NoiseResponse(0.5, 0.2, 1.0),
    ),
)
base = p.CalibratedFunctionalCausalRecipe(
    controls, nodes, ((0, 0),), ((1, 0),), calibration_rows=73
)
save(
    "environment.json",
    {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "source": p.__file__,
        "threads": torch.get_num_threads(),
        "seed": SEED,
        "task_id": TASK,
    },
)
save("recipe.json", asdict(base))
expected_cal = population(73, ("target-calibration-v1",))
expected_rows = population(19)
np.savez(OUT / "oracle-populations.npz", calibration=expected_cal, emitted=expected_rows)
for name, recipe, column in [
    ("downstream", base, 1),
    ("captured-converter", replace(base, target=((0, 1),)), 0),
]:
    task = p.generate_task(recipe, seed=SEED, task_id=TASK)
    card = json.loads(task.mechanism_json)
    cal = card["calibration"]
    raw = expected_cal[:, column : column + 1]
    mean = math.fsum(float(x) for x in raw[:, 0]) / len(raw)
    std = math.sqrt(math.fsum((float(x) - mean) ** 2 for x in raw[:, 0]) / len(raw))
    np.testing.assert_allclose(cal["center"], [mean], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(cal["scale"], [std], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        task.scores.numpy()[:, 0], (expected_rows[:, column] - mean) / std, rtol=2e-14, atol=2e-14
    )
    # Floating nonlinear evaluation differs in its last bits; exact digest oracle is
    # used only for the literal captured converter 2*x, whose order is exact here.
    if column == 0:
        digest = hashlib.sha256(b"73,1:")
        digest.update(np.ascontiguousarray(raw, dtype="<f8").tobytes())
        assert cal["sample_digest"] == digest.hexdigest()
    changed = p.generate_task(
        replace(recipe, task=replace(controls, rows=199, context_rows=99, split=81)),
        seed=SEED,
        task_id=TASK,
    )
    assert json.loads(changed.mechanism_json)["calibration"] == cal
    changed_labels = p.generate_task(
        replace(recipe, task=replace(controls, noise_ratio=2)), seed=SEED, task_id=TASK
    )
    assert json.loads(changed_labels.mechanism_json)["calibration"] == cal
    trace = p.explain_generation(task, [18, 0, 18, 7])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    save(name + "-mechanism.json", card)
    save(name + "-trace.json", trace)
    print("PASS independent scalar oracle:", name, "center", mean, "scale", std, flush=True)

for df, accepted in [(1.5, False), (2.0, False), (2.0001, True)]:
    heavy = replace(
        nodes[0],
        pipeline=None,
        exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=df)),
    )
    try:
        recipe = p.CalibratedFunctionalCausalRecipe(controls, (heavy,), ((0, 0),), ((0, 1),))
    except ValueError as err:
        if accepted or "finite second moments" not in str(err):
            raise AssertionError(str(err)) from err
    else:
        assert accepted
print("PASS strict second-moment boundary 1.5 / 2.0 rejected; 2.0001 accepted", flush=True)

for field, value in [
    ("rows", True),
    ("rows", 72),
    ("center", [True]),
    ("center", [None]),
    ("scale", [0]),
    ("scale", [float("inf")]),
    ("constant_channels", [False]),
    ("sample_digest", "z" * 64),
    ("stream", "wrong"),
    ("extra", 1),
]:
    task = p.generate_task(base, seed=SEED, task_id=TASK)
    card = json.loads(task.mechanism_json)
    card["calibration"][field] = value
    try:
        p.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0, 18])
    except ValueError:
        pass
    else:
        raise AssertionError(("malformed accepted", field, value))
print("PASS 10 malformed calibration metadata cases", flush=True)
required = base.admit()["estimated_working_bytes"]
p.functional_recipe_from_dict(
    json.loads(json.dumps(asdict(base))), limits=replace(base.limits, working_bytes=required)
)
try:
    p.functional_recipe_from_dict(
        json.loads(json.dumps(asdict(base))),
        limits=replace(base.limits, working_bytes=required - 1),
    )
except ValueError as err:
    if "memory" not in str(err):
        raise AssertionError(str(err)) from err
else:
    raise AssertionError("caller resource limit bypass")
try:
    replace(base, calibration_rows=65537)
except ValueError:
    pass
else:
    raise AssertionError("calibration max rows bypass")
print("PASS exact declared admission bound and row maximum", flush=True)

# Session create/save/open/replay with nonlinear/noisy converter graph, retained constants.
session = ExplorerSession(OUT / "session-original")
created = session.request(
    "generate", {"record": {"recipe": asdict(base), "seed": SEED, "task_id": TASK}}
)
graph = session.request("graph", {"indices": [18, 0, 18]})["graph"]
reopened = ExplorerSession(OUT / "session-original")
reopened.request("open", {"archive": created["archive"], "task_id": TASK})
assert reopened.request("graph", {"indices": [18, 0, 18]})["graph"] == graph
replay = ExplorerSession(OUT / "session-replay")
replay.request("generate", {"record_json": created["recipe_json"]})
assert replay.request("row", {"index": 18}) == session.request("row", {"index": 18})
save("session-result.json", created)
print("PASS public session create/save/open/replay", flush=True)

# A two-leaf literal graph induces subtraction overflow in normalization while
# both raw values and the correctly standardized scores remain finite.
leaves = ((-1.0, -1.7e308), (1.0, 1.7e308))
extreme_node = p.FunctionalCausalNode(
    (), p.FrozenTreeFunction(1, ((0,),), ((0.7,),), (leaves,)), root=p.JointRootBase("uniform", 1)
)
extreme = p.CalibratedFunctionalCausalRecipe(
    controls, (extreme_node,), ((0, 0),), ((0, 1),), calibration_rows=32
)
save("extreme-recipe.json", asdict(extreme))
task = p.generate_task(extreme, seed=SEED, task_id=TASK)
cal = json.loads(task.mechanism_json)["calibration"]
xcal = -1 + 2 * rng(("target-calibration-v1",), "root-base", 0).random((32, 1))[:, 0]
xout = -1 + 2 * rng((), "root-base", 0).random((19, 1))[:, 0]
scaledcal = np.where(xcal > 0.7, 1.0, -1.0)
scaledout = np.where(xout > 0.7, 1.0, -1.0)
assert np.any(scaledout != np.sign(cal["center"][0]))
np.testing.assert_allclose(
    task.scores.numpy()[:, 0],
    (scaledout - scaledcal.mean()) / scaledcal.std(),
    rtol=2e-14,
    atol=2e-14,
)
assert p.explain_generation(task, [0, 18])["signal_max_error"] == 0
save("extreme-mechanism.json", json.loads(task.mechanism_json))
print(
    "PASS finite extreme tree normalization including opposite-sign subtraction overflow",
    flush=True,
)
print("ALL BOUNDED PUBLIC CHECKS PASS", flush=True)
