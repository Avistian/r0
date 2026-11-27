"""Authored native starting point for the shared local and notebook workbench."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors


def native_example() -> dict[str, Any]:
    root = priors.NativeCausalNode(
        priors.FunctionalCausalNode(
            (), priors.FrozenLinearFunction(((1.0,),)), priors.JointRootBase("gaussian", 1)
        )
    )
    binary = priors.NativeCausalNode(
        priors.FunctionalCausalNode((0,), priors.FrozenLinearFunction(((1.0,),))),
        priors.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,)),
    )
    categorical = priors.VariableDomain("categorical", ("red", "green", "blue"), "softmax")
    category = priors.NativeCausalNode(
        priors.FunctionalCausalNode((0,), priors.FrozenLinearFunction(((-1.0,), (0.0,), (1.0,)))),
        categorical,
    )
    target = priors.NativeCausalNode(
        priors.FunctionalCausalNode(
            (2,), priors.FrozenLinearFunction(((1.0, 0.0, -1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 1.0)))
        ),
        categorical,
    )
    recipe = priors.NativeCausalRecipe(
        priors.LinearRecipe(
            rows=64,
            context_rows=48,
            features=3,
            signal_features=3,
            active_features=3,
            task_type="multiclass",
            classes=3,
        ),
        (root, binary, category, target),
        ((0, 0), (1, 0), (2, 0)),
        (3, 0),
    )
    controls = asdict(recipe)
    controls.pop("limits")
    return {"topology": "native", **controls}
