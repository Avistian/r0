import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from r0.modules import priors as p

OUT = Path(__file__).resolve().parent
base_files = {
    str(f.relative_to(OUT / "v1-base")): f.read_bytes()
    for f in (OUT / "v1-base").rglob("*")
    if f.is_file()
}
new_files = {
    str(f.relative_to(OUT / "v1-candidate")): f.read_bytes()
    for f in (OUT / "v1-candidate").rglob("*")
    if f.is_file()
}
assert len(base_files) == 129
assert base_files == new_files
print(
    "PASS exact v1 bytes:",
    len(base_files),
    "files, 12 cases, four roots and three adapters",
    flush=True,
)
controls = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
root = p.FunctionalCausalNode(
    (), p.FrozenLinearFunction(((1.0,), (2.0,))), root=p.JointRootBase("uniform", 1)
)
recipe = p.CalibratedFunctionalCausalRecipe(
    controls, (root,), ((0, 0),), ((0, 1),), calibration_rows=64
)
task = p.generate_task(recipe, seed=73, task_id=18)
for scale in [1e300, 1e-300]:
    scaled = replace(
        recipe, nodes=(replace(root, function=p.FrozenLinearFunction(((1.0,), (scale,)))),)
    )
    result = p.generate_task(scaled, seed=73, task_id=18)
    np.testing.assert_allclose(result.scores.numpy(), task.scores.numpy(), rtol=3e-14, atol=3e-14)
print("PASS large/small finite linear normalization cancellation", flush=True)
# Coordinated altered outputs are intentionally allowed by the unsigned-provenance
# contract. Here they demonstrate that inspection applies retained constants,
# instead of silently recomputing the original calibration population.
card = json.loads(task.mechanism_json)
card["calibration"]["center"] = [10.0]
card["calibration"]["scale"] = [4.0]
scores = (2 * task.inputs - 10) / 4
altered = replace(task, mechanism_json=json.dumps(card), scores=scores, targets=scores[:, 0])
trace = p.explain_generation(altered, [0, 15, 0])
assert trace["signal_max_error"] == trace["target_max_error"] == 0
(OUT / "retained-alternative-trace.json").write_text(
    json.dumps(trace, sort_keys=True, indent=2) + "\n"
)
print("PASS inspection uses supplied retained constants, without refitting", flush=True)
# Moment certificate after squaring must demand df > 4, not merely df > 2.
for df, accept in [(4.0, False), (4.0001, True)]:
    noisy = replace(
        root,
        noise_scale=(0.0, 1.0),
        exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=df)),
    )
    sq = p.FunctionalCausalNode(
        (0,),
        p.FrozenQuadraticFunction(
            2, (0, 1), (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),)
        ),
    )
    try:
        p.CalibratedFunctionalCausalRecipe(controls, (noisy, sq), ((0, 0),), ((1, 0),))
    except ValueError as err:
        if accept or "finite second moments" not in str(err):
            raise AssertionError(str(err)) from err
    else:
        assert accept
print("PASS nonlinear propagated moment gate at df=4", flush=True)
# Independent per-channel empirical moments for multiclass readouts.
controls = replace(controls, task_type="multiclass", classes=3)
node = replace(root, function=p.FrozenLinearFunction(((1.0,), (2.0,), (-3.0,), (4.0,))))
recipe = p.CalibratedFunctionalCausalRecipe(
    controls, (node,), ((0, 0),), ((0, 1), (0, 2), (0, 3)), calibration_rows=64
)
task = p.generate_task(recipe, seed=73, task_id=18)
key = json.dumps(
    ["functional-causal-v1", 73, 18, ["target-calibration-v1", "root-base", 0]],
    sort_keys=True,
    separators=(",", ":"),
)
x = (
    -1
    + 2
    * np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16)).random((64, 1))[:, 0]
)
raw = x[:, None] * np.array([2.0, -3.0, 4.0])
card = json.loads(task.mechanism_json)
np.testing.assert_allclose(card["calibration"]["center"], raw.mean(0), atol=3e-14, rtol=3e-14)
np.testing.assert_allclose(card["calibration"]["scale"], raw.std(0), atol=3e-14, rtol=3e-14)
np.testing.assert_allclose(
    task.scores.numpy(),
    (task.inputs.numpy() * [2.0, -3.0, 4.0] - raw.mean(0)) / raw.std(0),
    atol=3e-14,
    rtol=3e-14,
)
assert p.explain_generation(task, [0, 15])["target_max_error"] == 0
print("PASS multiclass independently calibrated channels and adapter replay", flush=True)
print("ALL SUPPLEMENTAL CHECKS PASS", flush=True)
