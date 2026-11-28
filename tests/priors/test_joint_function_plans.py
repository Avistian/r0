"""Source function catalogs retain ordered slots, including duplicated families."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_joint_function_catalog_preserves_preset_expansion_and_duplicate_slots() -> None:
    prior = priors.JointFunctionPrior(functions="tabpfnv2,lin,lin,default")
    expected = (
        "mlp",
        "tree",
        "disc",
        "lin",
        "lin",
        "mlp",
        "tree",
        "disc",
        "lin",
        "quad",
        "gp",
        "em",
        "prod",
    )
    request = prior.request()
    assert request.name == "random_function_fct_type"
    assert len(set(request.choices)) == len(expected)
    assert tuple(prior.family(slot) for slot in request.choices) == expected
    cheap = prior.request(cheap=True)
    assert cheap.name == "cheap_random_function_fct_type"
    assert tuple(prior.family(slot, cheap=True) for slot in cheap.choices) == (
        "gp",
        "tree",
        "disc",
        "lin",
        "quad",
    )
    with pytest.raises(ValueError, match="function"):
        priors.JointFunctionPrior(functions="lin, tree")
    with pytest.raises(ValueError, match="function"):
        priors.JointFunctionPrior(functions="unknown")


def test_joint_linear_and_quadratic_plans_keep_source_matrix_geometry_and_bindings() -> None:
    for family, shape in (("lin", (3, 21, 1)), ("quad", (21, 21, 3))):
        context = priors.JointSamplingContext(np.random.default_rng(28))
        prior = priors.JointFunctionPrior(
            functions=family, matrix=priors.JointMatrixPrior(families=("gaussian",))
        )
        plan = priors.sample_joint_function(
            context, "mechanism", input_width=21, output_width=3, calibration_rows=32, prior=prior
        )
        assert plan.family == family
        matrix = plan.matrices[0]
        assert (matrix.rows, matrix.columns, matrix.batch) == shape
        assert matrix.operation == ("mechanism", "matrix")
        record = context.freeze()
        plan.validate(record)
        assert [event.operation for event in record.events] == [
            ("mechanism",),
            ("mechanism", "matrix"),
        ]
        assert record.categorical.draws[0].request.name == "random_function_fct_type"
        assert record.categorical.draws[1].request.name == "random_matrix_type"


def test_joint_tree_plan_keeps_rounded_tree_count_and_floored_depth_requests() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(29))
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=4,
        output_width=2,
        calibration_rows=32,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    parameters = dict(plan.parameters)
    assert type(parameters["trees"]) is int
    assert 1 <= parameters["trees"] <= 128
    assert type(parameters["depth"]) is int
    assert 1 <= parameters["depth"] <= 7
    record = context.freeze()
    assert [draw.request.name for draw in record.numerical.draws] == ["rt_n_trees", "rt_depth"]
    assert [draw.request.integer for draw in record.numerical.draws] == [
        "round",
        "source-clip-floor-v1",
    ]
    assert plan.matrices == ()
    plan.validate(record)


def test_joint_gp_plan_keeps_product_and_radial_named_paths_separate() -> None:
    seen: set[bool] = set()
    for seed in (0, 2):
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode="global")
        plan = priors.sample_joint_function(
            context,
            "gp",
            input_width=3,
            output_width=2,
            calibration_rows=32,
            prior=priors.JointFunctionPrior(functions="gp"),
        )
        values = dict(plan.parameters)
        product = values["product_kernel"]
        assert type(product) is bool
        seen.add(product)
        assert plan.matrices == ()
        record = context.freeze()
        names = [d.request.name for d in record.numerical.draws]
        if product:
            assert plan.weights is None
            assert names == ["gp_a"]
        else:
            assert plan.weights is not None
            assert (plan.weights.width, plan.weights.batch) == (3, 1)
            assert names == [
                "gp_a",
                "simple_random_weights_decay_rate",
                "simple_random_weights_std_scale",
                "rand_gen_gp_scale",
            ]
            assert 0.5 <= values["scale"] <= 10
        assert record.categorical.draws[-1].request.name == "generalized_gp_use_product_kernel"
        plan.validate(record)
    assert seen == {False, True}


def test_joint_discretization_plan_retains_single_degenerate_and_explicit_center_cases() -> None:
    cases = (
        (1, 1, None, 1, "single-fit-row-v1", False),
        (2, 256, None, 1, "sampled-degenerate-count-v1", True),
        (7, 2, None, 1, "sampled-degenerate-count-v1", True),
        (7, 1, 10, 7, "explicit-capped-count-v1", False),
    )
    for rows, cap, explicit, expected, rule, sampled in cases:
        context = priors.JointSamplingContext(np.random.default_rng(30))
        prior = priors.JointFunctionPrior(
            functions="disc",
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
            max_discretization_cardinality=cap,
            discretization_centers=explicit,
        )
        plan = priors.sample_joint_function(
            context, "disc", input_width=3, output_width=2, calibration_rows=rows, prior=prior
        )
        assert (plan.center_count, plan.center_rule) == (expected, rule)
        requests = [draw.request.name for draw in context.freeze().numerical.draws]
        assert requests == (["rand_discr_n_centers"] if sampled else []) + ["cat_disc_p"]
        assert 0.5 <= dict(plan.parameters)["distance_p"] <= 4
        assert (plan.matrices[0].rows, plan.matrices[0].columns) == (2, 3)
        plan.validate(context.freeze())


def test_joint_em_requests_precede_the_output_matrix_and_keep_source_component_width() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(31))
    plan = priors.sample_joint_function(
        context,
        "em",
        input_width=3,
        output_width=9,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(
            functions="em", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    values = dict(plan.parameters)
    assert type(values["components"]) is int
    assert 2 <= values["components"] <= 18
    assert (plan.matrices[0].rows, plan.matrices[0].columns, plan.matrices[0].batch) == (
        9,
        values["components"],
        1,
    )
    record = context.freeze()
    assert [d.request.name for d in record.numerical.draws] == [
        "random_em_components",
        "gen_em_assignment_std_factor",
        "gen_em_assignment_p_inv",
        "gen_em_assignment_q",
    ]
    assert [event.role for event in record.events] == [
        "family",
        "components",
        "spread_factor",
        "distance_p",
        "distance_q",
        "family",
    ]
    assert record.numerical.draws[0].request.high == 19
    plan.validate(record)


def test_joint_mlp_plan_samples_actual_stage_order_including_one_layer_flags() -> None:
    paths = {
        0: ("linear:0", "activation:0"),
        1: ("activation:0", "linear:0"),
        2: ("linear:0", "activation:0", "linear:1", "activation:1", "linear:2", "activation:2"),
        5: ("linear:0", "activation:0", "linear:1"),
        8: ("activation:0", "linear:0", "activation:1"),
    }
    for seed, expected in paths.items():
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode="global")
        plan = priors.sample_joint_function(
            context,
            "mlp",
            input_width=5,
            output_width=2,
            calibration_rows=32,
            prior=priors.JointFunctionPrior(
                functions="mlp", matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        )
        record = context.freeze()
        observed = tuple(
            dict.fromkeys(
                event.operation[-1] for event in record.events if len(event.operation) == 2
            )
        )
        assert observed == expected
        assert record.numerical.draws[0].request.name == "nn_width"
        assert record.numerical.draws[1].request.name == "nn_n_layers"
        assert plan.matrices[0].columns == 5
        assert plan.matrices[-1].rows == 2
        assert all(activation.selector == "standard" for activation in plan.activations)
        if seed in (0, 1, 8):
            assert len(plan.matrices) == 1
            assert (plan.matrices[0].rows, plan.matrices[0].columns) == (2, 5)
        plan.validate(record)


def test_joint_product_plan_finishes_left_cheap_subtree_before_right() -> None:
    seen: set[str] = set()
    for seed in range(25):
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
        plan = priors.sample_joint_function(
            context,
            "product",
            input_width=3,
            output_width=2,
            calibration_rows=4,
            prior=priors.JointFunctionPrior(
                functions="prod", matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        )
        left, right = plan.children
        assert left.cheap
        assert right.cheap
        assert left.operation == ("product", "left")
        assert right.operation == ("product", "right")
        assert (left.input_width, left.output_width, left.calibration_rows) == (3, 2, 4)
        assert (right.input_width, right.output_width, right.calibration_rows) == (3, 2, 4)
        seen.update((left.family, right.family))
        record = context.freeze()
        owners = tuple(dict.fromkeys(event.operation[1] for event in record.events[1:]))
        assert owners == ("left", "right")
        family_draws = [
            d
            for d in record.categorical.draws
            if d.request.name == "cheap_random_function_fct_type"
        ]
        assert len(family_draws) == 2
        plan.validate(record)
    assert seen == {"gp", "tree", "disc", "lin", "quad"}


def test_joint_function_plan_json_reconstructs_all_branches_and_rejects_coerced_objects() -> None:
    for family in ("mlp", "tree", "disc", "lin", "quad", "gp", "em", "prod"):
        context = priors.JointSamplingContext(np.random.default_rng(32), mode="local")
        plan = priors.sample_joint_function(
            context,
            "function",
            input_width=3,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(
                functions=family, matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        )
        payload = json.loads(json.dumps(asdict(plan)))
        restored = priors.joint_function_plan_from_dict(payload, decisions=context.freeze())
        assert restored == plan
        payload["prior"] = list(payload["prior"].items())
        with pytest.raises(ValueError, match="object"):
            priors.joint_function_plan_from_dict(payload, decisions=context.freeze())
        with pytest.raises(ValueError, match="limit"):
            priors.joint_function_plan_from_dict(
                json.loads(json.dumps(asdict(plan))),
                decisions=context.freeze(),
                limits=priors.CausalLimits(parameters=1),
            )


def test_joint_function_failed_branch_restores_existing_decisions_and_rng() -> None:
    for mode in ("local", "meta", "global"):
        rng = np.random.default_rng(33)
        context = priors.JointSamplingContext(rng, mode=mode)
        priors.sample_joint_activation(context, "existing", selector="standard")
        before, state = context.freeze(), copy.deepcopy(rng.bit_generator.state)
        with pytest.raises(ValueError, match="bounds"):
            priors.sample_joint_function(
                context,
                "retry",
                input_width=2,
                output_width=2,
                calibration_rows=8,
                prior=priors.JointFunctionPrior(functions="disc", max_discretization_cardinality=1),
            )
        assert context.freeze() == before
        assert rng.bit_generator.state == state
        accepted = priors.sample_joint_function(
            context,
            "retry",
            input_width=2,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(functions="tree"),
        )
        accepted.validate(context.freeze())


def test_joint_function_nested_budget_failure_does_not_commit_family_or_child_choices() -> None:
    rng = np.random.default_rng(34)
    context = priors.JointSamplingContext(rng, limits=priors.PreferenceLimits(max_draws=2))
    before, state = context.freeze(), copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="limit"):
        priors.sample_joint_function(
            context,
            "nested",
            input_width=2,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(
                functions="lin", matrix=priors.JointMatrixPrior(families=("importance",))
            ),
        )
    assert context.freeze() == before
    assert rng.bit_generator.state == state


def test_joint_function_json_counts_prior_numbers_and_rejects_changed_matrix_ownership() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(35))
    plan = priors.sample_joint_function(
        context,
        "linear",
        input_width=2,
        output_width=2,
        calibration_rows=8,
        prior=priors.JointFunctionPrior(
            functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    payload = json.loads(json.dumps(asdict(plan)))
    decisions = context.freeze()
    # Three function dimensions, two prior counts, three matrix dimensions,
    # and two Boolean selector scopes are ten raw numeric leaves.
    assert (
        priors.joint_function_plan_from_dict(
            payload, decisions=decisions, limits=priors.CausalLimits(parameters=10)
        )
        == plan
    )
    with pytest.raises(ValueError, match="limit"):
        priors.joint_function_plan_from_dict(
            payload, decisions=decisions, limits=priors.CausalLimits(parameters=9)
        )
    for path, value in (
        (("input_width",), True),
        (("input_width",), 3),
        (("version",), "joint-function-plan-v2"),
        (("matrices", 0, "operation"), ["linear", "orphan"]),
        (("matrices", 0, "prior", "extra"), 1),
        (("matrices", 0, "prior"), []),
    ):
        corrupted = copy.deepcopy(payload)
        target = corrupted
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="width|geometry|version|decision|unknown|object"):
            priors.joint_function_plan_from_dict(corrupted, decisions=decisions)
