"""Independent diagnostic exposure and retained numeric-count/boundary checks."""

import json
from dataclasses import asdict, replace

from r0.modules import priors as p

C = p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8)
L = p.FrozenLinearFunction
root = p.FunctionalCausalNode((), L(((1.0,),)), root=p.JointRootBase("uniform", 1))
saturated = p.CalibratedActivation(
    kind="tanh",
    input_mean=(0.0,),
    input_inverse_std=(1.0,),
    shift=(100.0,),
    output_mean=(0.0,),
    output_inverse_std=(1.0,),
    shift_rows=(0,),
    calibration_rows=1,
    calibration_id="literal",
    calibration_digest="0" * 64,
)
mlp = p.FrozenMLPFunction((L(((1.0,),)),), (saturated,), end_activation=True)
warp = p.CalibratedTransform(
    kind="kumaraswamy",
    parameters=(("a", 1.0), ("b", 1.0)),
    minimum=(-0.01,),
    maximum=(0.01,),
    channels=1,
    calibration_rows=2,
    calibration_id="literal",
    calibration_digest="0" * 64,
)
for name, func, pipe, expected in [
    ("internal_mlp", mlp, None, "saturated_activation"),
    (
        "pipeline",
        L(((1.0,),)),
        p.CausalNodePipeline(1, stages=(p.CausalActivation("relu", (("input_shift", -2.0),)),)),
        "inactive_relu",
    ),
    (
        "converter_clamp",
        L(((1.0,),)),
        p.CausalNodePipeline(1, converters=(p.NumericalConverter(0, warp),)),
        "kumaraswamy_domain_clamp",
    ),
]:
    rec = p.FunctionalCausalRecipe(
        C, (root, p.FunctionalCausalNode((0,), func, pipeline=pipe)), ((0, 0),), ((1, 0),)
    )
    task = p.generate_task(rec, seed=11, task_id=5)
    card = json.loads(task.mechanism_json)
    assert expected in card["flags"], (name, card["flags"])
    assert any(expected in d["flags"] and d["node"] == 1 for d in card["node_diagnostics"])
    print("PASS exposed diagnostic", name, expected, flush=True)

    # Independent JSON-leaf counter, not the production dataclass traversal.
    def count(v):
        if type(v) in (int, float, bool):
            return 1
        if isinstance(v, dict):
            return sum(count(x) for x in v.values())
        if isinstance(v, (tuple, list)):
            return sum(map(count, v))
        return 0

    admission = rec.admit()
    assert admission["parameters"] == count(asdict(rec))
    assert admission["nodes"] == 2
    assert admission["total_channels"] == 2
    assert admission["estimated_analysis_bytes"] == 256 * 2 * 1 + 256 * 2
    cost = admission["estimated_working_bytes"]
    replace(rec, limits=replace(rec.limits, working_bytes=cost))
    try:
        replace(rec, limits=replace(rec.limits, working_bytes=cost - 1))
    except ValueError as e:
        if "working memory" not in str(e):
            raise AssertionError(str(e)) from e
    else:
        raise AssertionError("REJECT working admission boundary missing")
    print(
        "PASS numeric count, ancestry budget and exact working-byte gate",
        name,
        admission,
        flush=True,
    )
print("PASS: MLP/pipeline/clamp diagnostics and aggregate resource arithmetic.")
