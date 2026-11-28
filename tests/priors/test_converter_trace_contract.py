"""Typed converter inspection uses the shared original-row graph envelope."""

import numpy as np

from r0.modules import priors as p


def test_converted_trace_header_and_repeated_original_rows() -> None:
    category = p.CategoricalConverter(
        "side", 0, 1, ("negative", "positive"), "neighbor_id", ((-1.0,), (1.0,))
    )
    node = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,), (-1.0,))), p.JointRootBase("gaussian", 1)
        ),
        (category,),
    )
    recipe = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (node,),
        (p.ConvertedReadout(0, capture="side"),),
        p.ConvertedReadout(0, channel=1),
    )
    task = p.generate_task(recipe, seed=41, task_id=9)
    selected = [15, 0, 15]
    trace = p.explain_generation(task, selected)
    assert trace["version"] == "generation-graph-v1"
    assert trace["episode_id"] == task.episode_id
    assert trace["rows"] == selected
    assert trace["component"] is None
    assert trace["component_count"] == 0
    target = next(n for n in trace["nodes"] if n["id"] == "target")
    np.testing.assert_array_equal(target["values"], task.targets[selected, None].numpy())
