"""Guarded response grids retain the observed width and post-root input mask."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("feature", [0, 1, 3])
def test_guarded_view_curve_uses_mask_and_extra_columns(tmp_path: Path, feature: int):
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    fn = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (0.1,)),)),)
    )
    base = p.NetworkRecipe(task, fn)
    view = p.NetworkInputView(base, replace(task, features=4, active_features=1), (1,))
    session = ExplorerSession(tmp_path)
    session.request("generate", {"recipe": asdict(view), "seed": 4, "task_id": 8})
    original = session.task
    assert original is not None
    result = session.request(
        "curve", {"feature": feature, "points": 7, "anchor_row": 3, "bounds": [-2, 2]}
    )["curve"]
    x1 = np.array(result["values"]) if feature == 1 else np.full(7, original.inputs[3, 1])
    np.testing.assert_allclose(result["scores"], (3 * x1 + 0.1)[:, None], atol=1e-14)
    assert session.task is original
    # View aggregate admission, including extra observed columns, stays guarded.
    with pytest.raises(ValueError, match="limit"):
        session.request("curve", {"feature": feature, "causal_limits": {"working_bytes": 1}})
    assert session.task is original
