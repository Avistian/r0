"""Independent NumPy equations exercise all eight functions under four root bases."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r0.modules import priors as p

OUT = Path(os.environ["R0_VERIFY_OUT"])
line = p.FrozenLinearFunction


def quad():
    return p.FrozenQuadraticFunction(1, (0,), (((1.0, 0.0), (0.0, 0.0)),))


activation = p.CalibratedActivation(
    kind="tanh",
    input_mean=(0.0,),
    input_inverse_std=(1.0,),
    shift=(0.0,),
    output_mean=(0.0,),
    output_inverse_std=(1.0,),
    shift_rows=(0,),
    calibration_rows=1,
    calibration_id="literal-identity-normalization",
    calibration_digest="0" * 64,
)
functions = [
    ("linear", line(((2.0,),)), lambda x: 2 * x),
    (
        "quadratic",
        p.FrozenQuadraticFunction(1, (0,), (((1.0, 1.0), (2.0, 2.0)),)),
        lambda x: x * x + 3 * x + 2,
    ),
    (
        "tree",
        p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((-2.0,), (3.0,)),)),
        lambda x: np.where(x > 0, 3.0, -2.0),
    ),
    (
        "rff",
        p.FrozenRFFFunction(((0.7,),), (0.2,), ((1.3,),)),
        lambda x: 1.3 * np.cos(0.7 * x + 0.2),
    ),
    (
        "discretization",
        p.FrozenDiscretizationFunction(((-1.0,), (1.0,)), line(((3.0,),))),
        lambda x: np.where(np.abs(x + 1) <= np.abs(x - 1), -3.0, 3.0),
    ),
    (
        "em",
        p.FrozenEMFunction(((-1.0,), (1.0,)), (1.0, 1.0), line(((-2.0, 3.0),))),
        lambda x: (
            (-2 * np.exp(-np.abs(x + 1)) + 3 * np.exp(-np.abs(x - 1)))
            / (np.exp(-np.abs(x + 1)) + np.exp(-np.abs(x - 1)))
        ),
    ),
    ("product", p.FrozenProductFunction(quad(), line(((2.0,),))), lambda x: 2 * x**3),
    (
        "mlp",
        p.FrozenMLPFunction((line(((2.0,),)), line(((3.0,),))), (activation,)),
        lambda x: 3 * np.tanh(2 * x),
    ),
]
controls = p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8)
results = []
for family in ("uniform", "gaussian", "ball", "covariance"):
    base = (
        p.JointRootBase(family, 1)
        if family != "covariance"
        else p.JointRootBase(
            family,
            1,
            gaussian_factor=((3.0,),),
            weights=p.UniformWeightVector((0.5,)),
            factor=((1.5,),),
            covariance_base="gaussian",
        )
    )
    for name, func, oracle in functions:
        recipe = p.FunctionalCausalRecipe(
            controls,
            (
                p.FunctionalCausalNode((), line(((1.0,),)), root=base),
                p.FunctionalCausalNode((0,), func),
                p.FunctionalCausalNode((1,), quad()),
            ),
            ((0, 0),),
            ((2, 0),),
        )
        task = p.generate_task(
            p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=19, task_id=23
        )
        x = task.inputs.numpy()
        expected = oracle(x) ** 2
        np.testing.assert_allclose(task.scores.numpy(), expected, rtol=3e-14, atol=3e-14)
        trace = p.explain_generation(task, [15, 0, 15, 7])
        seen = set()
        for n in trace["nodes"]:
            assert set(n["parents"]) <= seen, (n["id"], n["parents"], seen)
            seen.add(n["id"])
        card = json.loads(task.mechanism_json)
        assert card["active"] == [0]
        assert card["moment_contract"]["target_orders"] == ["all"]
        results.append(
            {
                "base": family,
                "function": name,
                "max_absolute_error": float(np.max(np.abs(task.scores.numpy() - expected))),
                "signal_max_error": trace["signal_max_error"],
                "target_max_error": trace["target_max_error"],
                "topological": True,
            }
        )
        print(json.dumps(results[-1]), flush=True)
(OUT / "family_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(
    "PASS: 32 independent variable-input function equations, "
    "nonlinear descendants, JSON roundtrips and original-row traces."
)
