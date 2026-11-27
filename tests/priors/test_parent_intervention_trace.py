"""Effective parent slices retain sampled equations and construction ancestry."""

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors as p


def sample(width: int, seed: int):
    request = p.FeatureConv1DRequest(
        width,
        1,
        (
            p.Conv1DLayerPrior(
                (p.Conv1DGeometry(3, width),),
                kernel_laws=("uniform",),
                activations=(p.CausalActivation(),),
            ),
        ),
    )
    return p.sample_feature_conv1d(
        request, named_rng=np.random.default_rng(seed), raw_rng=np.random.default_rng(seed + 1)
    )


@pytest.mark.parametrize("joint", [False, True])
def test_effective_parent_children_and_joint_keep_raw_layers_and_input_slices(joint: bool):
    children = (sample(2, 41), sample(2, 51))
    joint_function = sample(4, 61) if joint else None
    replacement = p.FrozenParentFunction(children, "mean", joint=joint_function)
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(tuple(tuple(float(i == j) for j in range(2)) for i in range(2))),
        p.JointRootBase("gaussian", 2),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=4, active_features=4, signal_features=4)
    original = p.FrozenLinearFunction(((1.0, 0.0, 0.0, 0.0),) * 3)
    recipe = p.FunctionalCausalRecipe(
        task,
        (root, root, p.FunctionalCausalNode((0, 1), original)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((2, 0),),
    )
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.SoftIntervention(2, function=replacement),),
        seed=19,
        task_id=2**54 + 7,
    )
    trace = p.explain_generation(pair.after, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    inputs = pair.before.inputs.numpy()
    members: list[NDArray[np.float64]] = []
    for i, child in enumerate(children):
        layer = child.function.layers[0]
        values = inputs[:, 2 * i : 2 * i + 2]
        members.append(values @ np.asarray(layer.kernel)[:, 0, :].T + np.asarray(layer.bias))
        sliced = nodes[f"intervention-2-parent-input-{i}"]
        np.testing.assert_array_equal(sliced["values"], values[[15, 0, 15]])
        assert sliced["parents"] == ["intervention-2-inputs"]
    if joint_function is not None:
        layer = joint_function.function.layers[0]
        members.append(inputs @ np.asarray(layer.kernel)[:, 0, :].T + np.asarray(layer.bias))
    expected = np.mean(members, axis=0)
    np.testing.assert_allclose(pair.after.scores.numpy()[:, 0], expected[:, 0], rtol=0, atol=1e-12)
    raw = [n for n in nodes.values() if "construction-draw" in n["id"]]
    assert len(raw) == 5 * len(members)
    assert len(nodes["intervention-2-function"]["parents"]) == len(members)
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
    restored = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    replay = p.generate_task(restored, seed=19, task_id=2**54 + 7)
    assert p.explain_generation(replay, [15, 0, 15])["nodes"] == trace["nodes"]
