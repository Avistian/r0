"""Independent persisted-session equations and pinned legacy affine comparison."""

import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import r0.modules.priors.lib.causal_evaluation as ce
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession

OUT = Path(os.environ["R0_VERIFY_OUT"])
results = []
for family in ("uniform", "gaussian", "ball", "covariance"):
    base = (
        p.JointRootBase(family, 1)
        if family != "covariance"
        else p.JointRootBase(
            family,
            1,
            gaussian_factor=((2.0,),),
            weights=p.UniformWeightVector((0.5,)),
            factor=((1.0,),),
            covariance_base="uniform",
        )
    )
    for kind, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
        controls = p.LinearRecipe(
            rows=16,
            features=1,
            signal_features=1,
            active_features=1,
            context_rows=8,
            task_type=kind,
            classes=classes,
        )
        matrices = (((1.0, 0.0), (0.0, 0.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 1.0)))
        rec = p.FunctionalCausalRecipe(
            controls,
            (
                p.FunctionalCausalNode((), p.FrozenLinearFunction(((1.0,),)), root=base),
                p.FunctionalCausalNode((0,), p.FrozenQuadraticFunction(1, (0,), matrices)),
            ),
            ((0, 0),),
            tuple((1, i) for i in range(3 if kind == "multiclass" else 1)),
        )
        directory = OUT / "sessions" / f"{family}-{kind}"
        session = ExplorerSession(directory / "saved")
        first = session.request(
            "generate", {"record": {"recipe": dataclasses.asdict(rec), "seed": 91, "task_id": 4}}
        )
        trace = session.request("graph", {"indices": [15, 0, 15, 5]})["graph"]
        nodes = {n["id"]: n for n in trace["nodes"]}
        x = np.array(nodes["causal-0"]["values"])[:, 0]
        literal = np.column_stack((x * x, x, np.ones_like(x)))
        np.testing.assert_allclose(nodes["function-1"]["values"], literal, rtol=1e-14)
        seen = set()
        for n in trace["nodes"]:
            assert set(n["parents"]) <= seen
            seen.add(n["id"])
        reopened = ExplorerSession(directory / "saved")
        second = reopened.request("open", {"archive": first["archive"], "task_id": 4})
        assert second["view"] == first["view"]
        assert reopened.request("graph", {"indices": [15, 0, 15, 5]})["graph"] == trace
        replay = ExplorerSession(directory / "replayed")
        third = replay.request("generate", {"record_json": first["recipe_json"]})
        assert first["recipe_json"] == third["recipe_json"]
        assert replay.request("row", {"index": 15}) == session.request("row", {"index": 15})
        result = {
            "family": family,
            "task_type": kind,
            "archive": first["archive"],
            "literal_equation": "(x*x,x,1)",
            "signal_max_error": trace["signal_max_error"],
            "target_max_error": trace["target_max_error"],
            "save_open_replay": True,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
(OUT / "session_results.json").write_text(json.dumps(results, indent=2) + "\n")
# Compare old inline structural-noise arithmetic with the extracted shared helper,
# including generation's RNG/observation-label adapters and original-row tracing.
source = subprocess.check_output(
    [
        "git",
        "show",
        "de1bead86d51721fd84d84598b7bea2800ec8885:src/r0/modules/priors/lib/causal_stochastic.py",
    ],
    text=True,
)
(OUT / "baseline_causal_stochastic.py").write_text(source)
m = types.ModuleType("independent_pinned_affine_baseline")
sys.modules[m.__name__] = m
exec(compile(source, str(OUT / "baseline_causal_stochastic.py"), "exec"), m.__dict__)
current = ce.stochastic_value
cases = [
    (1.0, p.NoiseResponse(1.0, 0.2, 2.0)),
    (1e200, p.NoiseResponse(0.0, 1e-300, 2.0)),
    (1e-200, p.NoiseResponse(0.0, 1e300, 2.0)),
    (0.0, p.NoiseResponse(0.0, 1.0, 2.0)),
]
for bias, response in cases:
    node = p.StochasticCausalNode(bias=(bias,), noise_scale=(1.0,), noise_response=response)
    raw = torch.tensor([[-1.0], [0.0], [0.3], [1.0]], dtype=torch.float64)
    a = m.stochastic_value(node, raw, None, 0, None)
    b = current(node, raw, None, 0, None)
    assert all(x.numpy().tobytes() == y.numpy().tobytes() for x, y in zip(a, b, strict=True))
    print("PASS pinned affine scale bytes", bias, dataclasses.asdict(response), flush=True)
for kind, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
    for position in ("before_output", "after_output"):
        root = p.StochasticCausalNode(
            bias=(0.2,),
            noise_scale=(1.0,),
            exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=4.0)),
        )
        child = p.StochasticCausalNode(
            parents=(0,),
            weights=((1.0, -0.5, 0.3),),
            bias=(0.3, -0.1, 0.2),
            noise_scale=(0.2, 0.3, 0.4),
            noise_response=p.NoiseResponse(0.1, 0.3, 2.0),
            output_activation=p.CausalActivation("tanh"),
            noise_position=position,
        )
        rec = p.CausalRecipe(
            p.LinearRecipe(
                rows=16,
                features=1,
                signal_features=1,
                active_features=1,
                context_rows=8,
                task_type=kind,
                classes=classes,
            ),
            (root, child),
            ((0, 0),),
            tuple((1, i) for i in range(3 if kind == "multiclass" else 1)),
            version="causal-v2",
        )
        with patch.object(ce, "stochastic_value", m.stochastic_value):
            old = p.generate_task(rec, seed=37, task_id=9)
        new = p.generate_task(rec, seed=37, task_id=9)
        for field in dataclasses.fields(old):
            left, right = getattr(old, field.name), getattr(new, field.name)
            if isinstance(left, torch.Tensor):
                assert left.numpy().tobytes() == right.numpy().tobytes(), field.name
            else:
                assert left == right, field.name
        trace = p.explain_generation(new, [15, 0, 15])
        seen = set()
        for n in trace["nodes"]:
            assert set(n["parents"]) <= seen
            seen.add(n["id"])
        print(
            "PASS pinned legacy generation and topological inspection", kind, position, flush=True
        )
print(
    "PASS: 12 saved session cases; 4 affine extreme arithmetic cases; "
    "6 legacy generation/RNG cases."
)
