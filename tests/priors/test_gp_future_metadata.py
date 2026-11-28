"""Symbolic receipt accounting equals an independent traversal of actual GP draws."""

from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors as p


def literal_cost(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        children = [literal_cost(v) for pair in cast(dict[str, Any], value).items() for v in pair]
    elif isinstance(value, (tuple, list)):
        children = [literal_cost(v) for v in cast(list[Any], value)]
    elif value is None:
        return 0, 0, 1, 0
    elif isinstance(value, str):
        return 0, len(value.encode()), 1, 0
    else:
        return 1, 64, 1, 0
    return (
        sum(c[0] for c in children),
        32 + sum(c[1] for c in children),
        1 + sum(c[2] for c in children),
        1 + max((c[3] for c in children), default=-1),
    )


@pytest.mark.parametrize("target", ["linear", "quadratic", "neural"])
@pytest.mark.parametrize(("rows", "columns", "dimension", "kernels"), [(8, 2, 2, 1), (16, 3, 1, 3)])
def test_future_gp_metadata_is_exact(
    target: str,
    rows: int,
    columns: int,
    dimension: int,
    kernels: int,
    monkeypatch: pytest.MonkeyPatch,
):
    request = p.GPColumnRequest(
        rows=rows,
        columns=columns,
        dimension=dimension,
        kernels=tuple(
            p.GPColumnKernel(length_scale=0.03, amplitude=1 + i * 0.1) for i in range(kernels)
        ),
        target_family=target,
        outputs=2,
        hidden=4,
        calibration_rows=32,
    )
    actual = p.realize_gp_columns(request, seed=17, task_id=19)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("symbolic metadata cost performed covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    report = p.gp_column_population_metadata_cost(request)
    assert tuple(report[k] for k in ("numbers", "storage", "nodes", "depth")) == literal_cost(
        asdict(actual)
    )


def test_future_root_envelope_counts_actual_graph_and_bindings():
    request = p.GPColumnRequest(rows=16)
    source = p.realize_gp_columns(request, seed=17, task_id=19)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    node = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)))
    base = p.FunctionalCausalRecipe(task, (node, child), ((0, 0),), ((1, 0),))
    bindings = (p.GPRootBinding(0, (1,)),)
    actual = p.GPRootDAGRecipe(source, base, bindings)
    report = p.gp_root_dag_metadata_cost(request, base, bindings, limits=p.CausalLimits())
    assert tuple(report[k] for k in ("numbers", "storage", "nodes", "depth")) == literal_cost(
        asdict(actual)
    )
