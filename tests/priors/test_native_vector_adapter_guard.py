"""Native vector targets cannot silently accept label-adapter controls."""

import json
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch

from r0.modules import priors as p


def native(regression: bool = False) -> p.NativeVectorRecipe:
    return p.NativeVectorRecipe(
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            signal_features=1,
            active_features=1,
            task_type="regression" if regression else "binary",
            classes=0 if regression else 2,
        ),
        (
            p.NativeVectorNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,), (2.0,))), p.JointRootBase("gaussian", 1)
                ),
                (
                    p.VariableDomain(),
                    p.VariableDomain()
                    if regression
                    else p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,)),
                ),
            ),
        ),
        ((0, 0),),
        (0, 1),
    )


@pytest.mark.parametrize(
    ("field", "value"), [("noise_ratio", 100.0), ("separation", 2.0), ("class_offsets", (1.0,))]
)
@pytest.mark.parametrize("archive", [False, True])
def test_nondefault_native_adapter_controls_reject(field: str, value: Any, archive: bool):
    base = native(regression=field == "noise_ratio")
    if archive:
        raw = json.loads(json.dumps(asdict(base)))
        raw["task"][field] = value
        with pytest.raises(ValueError, match="adapter controls"):
            p.recipe_from_dict(raw)
    else:
        task = replace(base.task, **{field: value})
        with pytest.raises(ValueError, match="adapter controls"):
            replace(base, task=task)


def test_default_native_target_retains_direct_threshold_semantics():
    task = p.generate_task(native(), seed=13, task_id=17)
    torch.testing.assert_close(task.targets, (task.inputs[:, 0] >= 0).long(), rtol=0, atol=0)
    assert task.target_origin == "native-variable-v1"
