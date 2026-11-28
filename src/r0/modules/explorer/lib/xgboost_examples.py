"""Editable shared-vector XGBoost construction plans with explicit output semantics."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def xgboost_examples() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for objective in ("reg:squarederror", "binary:logistic", "multi:softprob"):
        skeleton = p.FunctionalCausalRecipe(
            p.LinearRecipe(
                rows=64, context_rows=48, features=2, signal_features=2, active_features=2
            ),
            (
                p.FunctionalCausalNode(
                    (),
                    p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
                    p.JointRootBase("gaussian", 2),
                ),
                p.FunctionalCausalNode(
                    (0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0), (1.0, -1.0)))
                ),
            ),
            ((0, 0), (0, 1)),
            ((1, 0),),
        )
        result[f"XGBoost shared-vector fit: {objective}"] = {
            "topology": "xgboost-fitted",
            "skeleton": asdict(skeleton),
            "fits": [
                {
                    "node": 1,
                    "plan": asdict(
                        p.XGBoostFitPlan(
                            input_width=2,
                            output_width=3,
                            rows=64,
                            estimators=3,
                            depth=2,
                            max_bin=32,
                            objective=objective,
                        )
                    ),
                }
            ],
        }
    return result
