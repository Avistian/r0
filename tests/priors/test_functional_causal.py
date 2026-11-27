"""Functional graphs remain nonlinear through generation, persistence and inspection."""

import json
from dataclasses import asdict

import torch

from r0.modules.priors import (
    FrozenQuadraticFunction,
    FrozenTreeFunction,
    FunctionalCausalNode,
    FunctionalCausalRecipe,
    JointRootBase,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def test_dimension_changing_tree_root_feeds_quadratic_and_replays() -> None:
    # A one-dimensional root produces (2,3). The child computes x^2 + 3y^2 = 31.
    root = FunctionalCausalNode(
        parents=(),
        root=JointRootBase("uniform", 1),
        function=FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0, 3.0), (2.0, 3.0)),)),
    )
    child = FunctionalCausalNode(
        parents=(0,),
        function=FrozenQuadraticFunction(
            2, (0, 1), (((1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.0)),)
        ),
    )
    recipe = FunctionalCausalRecipe(
        task=LinearRecipe(
            rows=16, features=2, signal_features=2, active_features=2, context_rows=8
        ),
        nodes=(root, child),
        observed=((0, 0), (0, 1)),
        target=((1, 0),),
    )
    decoded = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    task = generate_task(decoded, seed=17, task_id=9)
    assert torch.equal(task.inputs, torch.tensor([[2.0, 3.0]] * 16, dtype=torch.float64))
    assert torch.equal(task.scores, torch.full((16, 1), 31.0, dtype=torch.float64))
    trace = explain_generation(task, [15, 2, 15])
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert nodes["function-1"]["values"] == [[31.0], [31.0], [31.0]]
    assert nodes["function-1"]["parents"] == ["causal-0"]
    assert trace["signal_max_error"] == trace["target_max_error"] == 0


def test_nonlinear_noise_response_uses_function_output_before_converter_and_importance() -> None:
    from r0.modules.priors import (
        CausalNodePipeline,
        FrozenLinearFunction,
        NoiseResponse,
        NumericalConverter,
    )

    root = FunctionalCausalNode(
        (),
        FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0,), (2.0,)),)),
        root=JointRootBase("gaussian", 1),
    )
    child = FunctionalCausalNode(
        (0,),
        FrozenQuadraticFunction(1, (0,), (((1.0, 0.0), (0.0, 0.0)),)),
        noise_scale=(2.0,),
        noise_response=NoiseResponse(offset=1.0, strength=3.0),
        pipeline=CausalNodePipeline(
            1,
            channel_importance=(2.0,),
            node_importance=3.0,
            converters=(NumericalConverter(0),),
        ),
    )
    descendant = FunctionalCausalNode((1,), FrozenLinearFunction(((1.0,),)))
    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (root, child, descendant),
        ((1, 0),),
        ((2, 0),),
    )
    task = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=3, task_id=4
    )
    assert torch.equal(task.inputs, torch.full((16, 1), 8.0, dtype=torch.float64))
    trace = explain_generation(task, [7, 1, 7])
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert nodes["noise-scale-1"]["values"] == [[26.0]] * 3
    raw = torch.tensor(nodes["exogenous-1"]["values"], dtype=torch.float64)
    torch.testing.assert_close(task.scores[[7, 1, 7]], 24.0 + 78.0 * raw)
    assert nodes["readout-1"]["values"] == [[8.0]] * 3
    assert nodes["readout-1"]["parents"] == ["pipeline-1-converted-readout"]
    assert nodes["pipeline-1-channel-importance"]["values"] == [[8.0]] * 3
    assert trace["signal_max_error"] == trace["target_max_error"] == 0


def test_finite_calibration_cannot_certify_heavy_tailed_parent_normalization() -> None:
    import numpy as np
    import pytest

    from r0.modules.priors import (
        CausalActivation,
        ExogenousDistribution,
        FrozenLinearFunction,
        FrozenMLPFunction,
        InputDistribution,
        calibrate_source_activation,
    )

    frozen = calibrate_source_activation(
        CausalActivation("relu"),
        np.array([[-1.0], [1.0]]),
        rng=np.random.default_rng(1),
        scale=1.0,
        calibration_id="independent-fixture",
    )
    root = FunctionalCausalNode(
        (),
        FrozenLinearFunction(((1.0,),)),
        root=JointRootBase("uniform", 1),
        noise_scale=(1.0,),
        exogenous=ExogenousDistribution(InputDistribution("student_t", df=1.5)),
    )
    child = FunctionalCausalNode(
        (0,),
        FrozenMLPFunction((FrozenLinearFunction(((1.0,),)),), (frozen,), start_activation=True),
    )
    with pytest.raises(ValueError, match="second moments"):
        FunctionalCausalRecipe(
            LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
            (root, child),
            ((0, 0),),
            ((1, 0),),
        )


def test_all_eight_families_compose_with_literal_outputs() -> None:
    import math

    from r0.modules import priors

    root = FunctionalCausalNode(
        (),
        FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0,), (2.0,)),)),
        root=JointRootBase("ball", 1),
    )
    functions: tuple[priors.FrozenFunction, ...] = (
        priors.FrozenLinearFunction(((3.0,),)),
        FrozenQuadraticFunction(1, (0,), (((1.0, 0.0), (0.0, 0.0)),)),
        FrozenTreeFunction(1, ((0,),), ((1.0,),), (((5.0,), (7.0,)),)),
        priors.FrozenRFFFunction(((math.pi / 2,),), (0.0,), ((2.0,),)),
        priors.FrozenDiscretizationFunction(
            ((1.0,), (3.0,)), priors.FrozenLinearFunction(((2.0,),))
        ),
        priors.FrozenEMFunction(
            ((2.0,), (2.0,)), (1.0, 2.0), priors.FrozenLinearFunction(((1.0, 0.0),))
        ),
        priors.FrozenProductFunction(
            priors.FrozenLinearFunction(((2.0,),)), priors.FrozenLinearFunction(((3.0,),))
        ),
        priors.FrozenMLPFunction(
            (priors.FrozenLinearFunction(((2.0,),)), priors.FrozenLinearFunction(((3.0,),))),
            (
                priors.CalibratedActivation(
                    kind="identity",
                    input_mean=(0.0,),
                    input_inverse_std=(1.0,),
                    shift=(0.0,),
                    output_mean=(0.0,),
                    output_inverse_std=(1.0,),
                    shift_rows=(0,),
                    calibration_rows=1,
                    calibration_id="literal-identity",
                    calibration_digest="0" * 64,
                ),
            ),
        ),
    )
    total = FunctionalCausalNode(tuple(range(1, 9)), priors.FrozenLinearFunction(((1.0,) * 8,)))
    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=8, signal_features=8, active_features=8, context_rows=8),
        (root, *(FunctionalCausalNode((0,), f) for f in functions), total),
        tuple((i, 0) for i in range(1, 9)),
        ((9, 0),),
    )
    task = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=1, task_id=2
    )
    expected = torch.tensor([[6, 4, 7, -2, 2, 2 / 3, 24, 12]] * 16, dtype=torch.float64)
    torch.testing.assert_close(task.inputs, expected)
    torch.testing.assert_close(task.scores, torch.full((16, 1), 53 + 2 / 3, dtype=torch.float64))
    trace = explain_generation(task, [15, 0])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    card = json.loads(task.mechanism_json)
    assert card["moment_contract"]["target_orders"] == ["all"]
    assert "zero_target_dispersion" in card["flags"]


def test_functional_json_rejects_object_instead_of_activation_parameter_array() -> None:
    import pytest

    from r0.modules import priors

    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (
            FunctionalCausalNode(
                (),
                priors.FrozenLinearFunction(((1.0,), (2.0,))),
                root=JointRootBase("uniform", 1),
                pipeline=priors.CausalNodePipeline(2, stages=(priors.CausalActivation(),)),
            ),
        ),
        ((0, 0),),
        ((0, 1),),
    )
    payload = json.loads(json.dumps(asdict(recipe)))
    payload["nodes"][0]["pipeline"]["stages"][0]["parameters"] = {}
    with pytest.raises(ValueError, match="array"):
        recipe_from_dict(payload)


def test_one_node_limit_counts_graph_nodes_not_function_dimensions() -> None:
    from r0.modules import priors

    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (
            FunctionalCausalNode(
                (), priors.FrozenLinearFunction(((1.0,), (2.0,))), root=JointRootBase("uniform", 1)
            ),
        ),
        ((0, 0),),
        ((0, 1),),
        limits=priors.CausalLimits(nodes=1),
    )
    decoded = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    task = generate_task(decoded, seed=5, task_id=6)
    torch.testing.assert_close(task.scores, 2 * task.inputs)


def test_shared_input_product_uses_dependent_moment_bound() -> None:
    from r0.modules import priors

    root = FunctionalCausalNode(
        (),
        priors.FrozenLinearFunction(((1.0,),)),
        root=JointRootBase("uniform", 1),
        noise_scale=(1.0,),
        exogenous=priors.ExogenousDistribution(priors.InputDistribution("student_t", df=4.0)),
    )
    child = FunctionalCausalNode(
        (0,),
        priors.FrozenProductFunction(
            FrozenQuadraticFunction(1, (0,), (((1.0, 0.0), (0.0, 0.0)),)),
            priors.FrozenLinearFunction(((1.0,),)),
        ),
    )
    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (root, child),
        ((0, 0),),
        ((1, 0),),
    )
    task = generate_task(recipe, seed=3, task_id=4)
    card = json.loads(task.mechanism_json)
    # The child is X^3, where X contains t_4 noise. It has moments only below 4/3.
    assert card["moment_contract"]["target_orders"] == [4 / 3]
    assert card["active"] == [0]
    torch.testing.assert_close(task.scores, task.inputs**3)


def test_aggregate_graph_limits_and_structural_contracts() -> None:
    from dataclasses import replace

    import pytest

    from r0.modules import priors

    root = FunctionalCausalNode(
        (), priors.FrozenLinearFunction(((1.0,),)), root=JointRootBase("uniform", 1)
    )
    child = FunctionalCausalNode((0,), priors.FrozenLinearFunction(((2.0,),)))
    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (root, child),
        ((0, 0),),
        ((1, 0),),
    )
    for limits in (
        priors.CausalLimits(nodes=1),
        priors.CausalLimits(total_channels=1),
        priors.CausalLimits(parameters=1),
        priors.CausalLimits(working_bytes=1024),
    ):
        with pytest.raises(ValueError, match="limit"):
            replace(recipe, limits=limits)
    with pytest.raises(ValueError, match="parent-before-child"):
        replace(recipe, nodes=(root, replace(child, parents=(1,))))
    with pytest.raises(ValueError, match="parent state widths"):
        replace(
            recipe,
            nodes=(root, replace(child, function=priors.FrozenLinearFunction(((1.0, 2.0),)))),
        )
    with pytest.raises(ValueError, match="target channel"):
        replace(recipe, observed=((1, 0),))
    with pytest.raises(ValueError, match="root base width"):
        replace(root, root=JointRootBase("uniform", 2))


def test_constant_internal_function_is_retained_and_flagged() -> None:
    from r0.modules import priors

    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (
            FunctionalCausalNode(
                (), priors.FrozenLinearFunction(((1.0,), (2.0,))), root=JointRootBase("uniform", 1)
            ),
            FunctionalCausalNode((0,), priors.FrozenLinearFunction(((0.0, 0.0),))),
        ),
        ((0, 0),),
        ((0, 1),),
    )
    task = generate_task(recipe, seed=1, task_id=2)
    card = json.loads(task.mechanism_json)
    assert "constant_function_sample" in card["flags"]
    assert any(
        d["node"] == 1 and d["stage"] == "function" and d["constant_channels"] == [0]
        for d in card["node_diagnostics"]
    )
    trace = explain_generation(task, [0, 1, 2])
    assert next(n for n in trace["nodes"] if n["id"] == "causal-1")["values"] == [[0.0]] * 3


def test_mlp_and_pipeline_saturation_diagnostics_reach_task_card() -> None:
    from r0.modules import priors

    activation = priors.CalibratedActivation(
        kind="tanh",
        input_mean=(0.0,),
        input_inverse_std=(1.0,),
        shift=(0.0,),
        output_mean=(0.0,),
        output_inverse_std=(1.0,),
        shift_rows=(0,),
        calibration_rows=1,
        calibration_id="authored-tanh",
        calibration_digest="0" * 64,
    )
    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (
            FunctionalCausalNode(
                (),
                FrozenTreeFunction(1, ((0,),), ((0.0,),), (((100.0,), (100.0,)),)),
                root=JointRootBase("uniform", 1),
            ),
            FunctionalCausalNode(
                (0,),
                priors.FrozenMLPFunction(
                    (priors.FrozenLinearFunction(((1.0,),)),), (activation,), end_activation=True
                ),
            ),
            FunctionalCausalNode(
                (0,),
                priors.FrozenLinearFunction(((1.0,),)),
                pipeline=priors.CausalNodePipeline(1, stages=(priors.CausalActivation("sigmoid"),)),
            ),
        ),
        ((1, 0),),
        ((2, 0),),
    )
    card = json.loads(generate_task(recipe, seed=1, task_id=2).mechanism_json)
    assert "saturated_activation" in card["flags"]
    saturated = {
        (d["node"], d["kind"])
        for d in card["node_diagnostics"]
        if "saturated_activation" in d["flags"]
    }
    assert saturated == {(1, "tanh"), (2, "sigmoid")}


def test_functional_trace_is_parent_before_child_for_renderer() -> None:
    from r0.modules import priors

    recipe = FunctionalCausalRecipe(
        LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
        (
            FunctionalCausalNode(
                (), priors.FrozenLinearFunction(((1.0,), (2.0,))), root=JointRootBase("uniform", 1)
            ),
        ),
        ((0, 0),),
        ((0, 1),),
    )
    trace = explain_generation(generate_task(recipe, seed=1, task_id=2), [0, 15])
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen, node["id"]
        seen.add(node["id"])


def test_existing_causal_traces_also_place_readouts_before_observations() -> None:
    from r0.modules import priors

    for version in ("causal-v1", "causal-v2"):
        recipe = priors.CausalRecipe(
            LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
            (priors.CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),),
            ((0, 0),),
            ((0, 1),),
            version=version,
        )
        trace = explain_generation(generate_task(recipe, seed=1, task_id=2), [0, 15])
        seen: set[str] = set()
        for node in trace["nodes"]:
            assert set(node["parents"]) <= seen, (version, node["id"])
            seen.add(node["id"])
