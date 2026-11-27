"""Untrusted native numerical fields reject before NumPy sees nested values."""

import json
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("field", ["base", "split", "leaf"])
def test_nested_native_numbers_reject_before_numpy_allocation(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = p.fit_xgboost(
        p.XGBoostFitPlan(rows=32, estimators=2, output_width=2),
        input_rng=np.random.default_rng(1),
        target_rng=np.random.default_rng(2),
        backend_rng=np.random.default_rng(3),
    )
    data = asdict(fit)
    native = json.loads(fit.model_json)
    tree = native["learner"]["gradient_booster"]["model"]["trees"][0]
    if field == "base":
        native["learner"]["learner_model_param"]["base_score"] = json.dumps([[0.0] * 10000])
    elif field == "split":
        tree["split_conditions"][0] = [0.0] * 10000
    else:
        tree["leaf_weights"][0] = [0.0] * 10000
    data["model_json"] = json.dumps(native)
    original_array = np.asarray
    original_scalar = np.float32
    calls: list[int] = []

    def observed_array(*args: Any, **kwargs: Any) -> Any:
        result = original_array(*args, **kwargs)
        if result.size > 1000:
            calls.append(result.size)
        return result

    class ObservedFloat32(np.float32):
        def __new__(cls, value: Any = 0) -> Any:
            if isinstance(value, (list, tuple)):
                calls.append(len(cast(list[Any] | tuple[Any, ...], value)))
                raise AssertionError("nested value reached numerical scalar conversion")
            return original_scalar(value)

    monkeypatch.setattr(np, "asarray", observed_array)
    monkeypatch.setattr(np, "float32", ObservedFloat32)
    with pytest.raises(ValueError, match="native|parameter|vector"):
        p.xgboost_fit_from_dict(data, limits=p.CausalLimits(parameters=5000))
    assert calls == []
