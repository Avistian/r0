"""Network internals remain inspectable inside joint native vector mechanisms."""

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("root_network", [True, False])
def test_vector_network_original_rows_and_intervention_trace(root_network: bool):
    network = p.FrozenNetworkFunction(
        1,
        (p.NetworkBlock((0,), (p.NetworkAffine(((2.0,), (-3.0,)), (0.5, -0.25)),)),),
    )
    continuous = p.VariableDomain()
    binary = p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))
    node = p.NativeVectorNode(
        p.FunctionalCausalNode(
            () if root_network else (0,),
            network,
            p.JointRootBase("gaussian", 1) if root_network else None,
        ),
        (continuous, binary),
    )
    index = 0 if root_network else 1
    nodes = (
        (node,)
        if root_network
        else (
            p.NativeVectorNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
                ),
                (continuous,),
            ),
            node,
        )
    )
    recipe = p.NativeVectorRecipe(
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            signal_features=1,
            active_features=1,
            task_type="binary",
            classes=2,
        ),
        nodes,
        ((index, 0),),
        (index, 1),
    )
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.NativeVectorHardIntervention(index, 0, value=7.0),),
        seed=41,
        task_id=9,
    )
    for task in (pair.before, pair.after):
        trace = p.explain_generation(task, [15, 0, 15])
        by_id = {n["id"]: n for n in trace["nodes"]}
        assert len(by_id) == len(trace["nodes"])
        assert all(parent in by_id for n in trace["nodes"] for parent in n["parents"])
        source = np.asarray(by_id[f"network-{index}-input"]["values"])[:, 0]
        expected = np.stack((2 * source + 0.5, -3 * source - 0.25), axis=1)
        np.testing.assert_allclose(
            by_id[f"network-{index}-block-1-state"]["values"],
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_array_equal(task.targets[[15, 0, 15]], expected[:, 1] >= 0)
