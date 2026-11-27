"""Mixed convolution/XGBoost authored drafts preserve strict archived schemas."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def mixed_recipe() -> p.FunctionalCausalRecipe:
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=24, context_rows=11, features=2, active_features=2, signal_features=2),
        (
            p.FunctionalCausalNode(
                (),
                p.FrozenLinearFunction(
                    tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
                ),
                p.JointRootBase("gaussian", 4),
            ),
            p.FunctionalCausalNode(
                (0,),
                p.FrozenFeatureConv1D(
                    4,
                    1,
                    (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((1.0, -2.0),),), (0.25,)),),
                ),
            ),
            p.FunctionalCausalNode(
                (1,),
                p.FrozenXGBoostVector(
                    3,
                    (0.25,),
                    (
                        p.XGBoostVectorTree(
                            (0, -1, -1),
                            (0.5, 0.0, 0.0),
                            (1, -1, -1),
                            (2, -1, -1),
                            (True, False, False),
                            ((0.0,), (1.0,), (-3.0,)),
                        ),
                    ),
                ),
            ),
        ),
        ((0, 0), (0, 3)),
        ((2, 0),),
    )


def browser_numbers(value: Any) -> Any:
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: browser_numbers(item) for key, item in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [browser_numbers(item) for item in cast(list[Any], value)]
    return value


def test_mixed_convolution_draft_real_numbers_and_strict_archive(tmp_path: Path) -> None:
    recipe = mixed_recipe()
    session = ExplorerSession(tmp_path)
    baseline = session.request(
        "generate",
        {
            "record_json": json.dumps(
                {"recipe": asdict(recipe), "seed": str(2**53 + 5), "task_id": str(2**53 + 7)}
            )
        },
    )
    draft = browser_numbers(session.request("mechanism", {})["editable_recipe"])
    draft["nodes"][1]["function"]["layers"][0]["bias"] = [0.75]
    with pytest.raises(ValueError, match="invalid vector tree field"):
        p.recipe_from_dict(draft)
    validated = session.request("convolution_draft", {"recipe": draft})["editable_recipe"]
    restored = p.recipe_from_dict(validated)
    assert isinstance(restored, p.FunctionalCausalRecipe)
    assert restored.nodes[2].function == recipe.nodes[2].function
    assert session.request("view", {}) == baseline
    # Deliberate integral real edits are valid; integer indices and bools are not retyped.
    changed = deepcopy(draft)
    changed["nodes"][2]["function"]["trees"][0]["values"][1] = [2]
    changed["nodes"][2]["function"]["trees"][0]["thresholds"][0] = 1
    edited = session.request("convolution_draft", {"recipe": changed})["editable_recipe"]
    assert edited["nodes"][2]["function"]["trees"][0]["values"][1] == [2.0]
    for field, value in (("thresholds", True), ("features", 0.0)):
        invalid = deepcopy(changed)
        invalid["nodes"][2]["function"]["trees"][0][field][0] = value
        with pytest.raises(ValueError, match="Convolution inspection failed"):
            session.request("convolution_draft", {"recipe": invalid})
        assert session.request("view", {}) == baseline
