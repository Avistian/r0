"""Mixed categorical captures never hide numerical readouts or descendant states."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


def graph(family: str, target: str = "numeric") -> p.RecursivePopulationGraph:
    controls = p.LinearRecipe(
        rows=16, context_rows=7, features=2, active_features=2, signal_features=2
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    mixed = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))))
    child = p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((2.0, 3.0),)))
    if family == "converted":
        converter = p.CategoricalConverter(
            "kind", 0, 1, ("low", "high"), "neighbor_disc", ((-2.0,), (3.0,))
        )
        observed = (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1))
        selected = p.ConvertedReadout(1, channel=1)
        if target == "capture":
            controls = replace(controls, task_type="binary", classes=2)
            selected = p.ConvertedReadout(1, capture="kind")
        elif target == "descendant":
            observed = (p.ConvertedReadout(1, capture="kind"), p.ConvertedReadout(1, channel=1))
            selected = p.ConvertedReadout(2, channel=0)
        base = p.ConvertedCausalRecipe(
            controls,
            (
                p.ConvertedCausalNode(root),
                p.ConvertedCausalNode(mixed, (converter,)),
                p.ConvertedCausalNode(child),
            ),
            observed,
            selected,
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            controls,
            tuple(p.NativeCausalNode(n) for n in (root, mixed, child)),
            ((1, 0), (1, 1)),
            (2, 0),
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            controls,
            tuple(
                p.NativeVectorNode(n, (p.VariableDomain(),) * n.function.output_width)
                for n in (root, mixed, child)
            ),
            ((1, 0), (1, 1)),
            (2, 0),
        )
    else:
        base = p.FunctionalCausalRecipe(controls, (root, mixed, child), ((1, 0), (1, 1)), ((2, 0),))
    plan = p.RecursiveForestPlan(
        input_width=2,
        estimators=2,
        min_depth=1,
        max_depth=1,
        leaf_mean=(1.0, -2.0),
        leaf_scale=(0.5, 2.0),
        source="r0-extension-v1",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    return p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=31,
        task_id=37,
        rng=np.random.default_rng(41),
    )


@pytest.mark.parametrize(
    ("family", "target"),
    [
        ("converted", "numeric"),
        ("converted", "capture"),
        ("converted", "descendant"),
        ("functional", "descendant"),
        ("native", "descendant"),
        ("vector", "descendant"),
    ],
)
def test_readouts_captures_and_descendants_are_topologically_complete(family: str, target: str):
    recipe = graph(family, target)
    task = p.generate_task(recipe, seed=31, task_id=37)
    rows = [12, 1, 12]
    trace = p.explain_generation(task, rows)
    nodes = {n["id"]: n for n in trace["nodes"]}
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert node["id"] not in seen
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
    np.testing.assert_array_equal(np.asarray(nodes["target"]["values"])[:, 0], task.targets[rows])
    assert nodes["dag-2-inputs"]["parents"] == ["dag-1-state"]
    np.testing.assert_array_equal(nodes["dag-2-inputs"]["values"], nodes["dag-1-state"]["values"])
    if family == "converted":
        readout = np.asarray(nodes["dag-1-readout"]["values"])
        capture = np.asarray(nodes["dag-1-capture:kind:value"]["values"])[:, 0]
        assert {"dag-1-readout", "dag-1-capture:kind:value"} <= set(nodes["dag-1-state"]["parents"])
        if target == "numeric":
            np.testing.assert_array_equal(readout[:, 1], task.targets[rows])
        elif target == "capture":
            np.testing.assert_array_equal(capture, task.targets[rows])
        else:
            np.testing.assert_array_equal(capture, task.inputs[rows, 0])
            np.testing.assert_array_equal(readout[:, 1], task.inputs[rows, 1])
    if target == "descendant":
        state = np.asarray(nodes["dag-1-state"]["values"])
        np.testing.assert_allclose(
            task.targets[rows], 2 * state[:, 0] + 3 * state[:, 1], rtol=0, atol=1e-14
        )
