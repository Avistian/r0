"""Graph plans preserve dimension-changing parent order and typed empty owners."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize(
    "matrix", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
@pytest.mark.parametrize("root", ["uniform", "ball", "gaussian", "covariance"])
@pytest.mark.parametrize("function", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_graph_plan_binds_all_function_families_and_ordered_parent_widths(
    function: str,
    matrix: str,
    root: str,
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(43), mode="local")
    function_prior = priors.JointFunctionPrior(
        functions=function,
        frequencies=4,
        matrix=priors.JointMatrixPrior(families=(matrix,)),
    )
    plan = priors.sample_joint_graph_plan(
        context,
        "graph",
        parents=((), (), (1, 0)),
        widths=(2, 3, 1),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=(root,), function=function_prior),
        function_prior=function_prior,
    )
    registry = context.freeze_registry()
    plan.validate(registry)
    assert plan.nodes[2].parents == (1, 0)
    assert isinstance(plan.nodes[2].mechanism, priors.JointFunctionPlan)
    assert plan.nodes[2].mechanism.input_width == 5
    assert plan.nodes[2].mechanism.output_width == 1
    assert plan.nodes[2].mechanism.family == function
    assert tuple(owner.operation for owner in plan.owners()) == tuple(
        span.operation for span in registry.operations
    )
    assert plan.owners()[0].kind == "graph"
    assert plan.owners()[1].kind == "node"
    assert plan.owners()[2].kind == "root"


def test_graph_plan_rejects_foreign_empty_owner_even_without_named_events() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(43), mode="local")
    plan = priors.sample_joint_graph_plan(
        context,
        "graph",
        parents=((),),
        widths=(1,),
        calibration_rows=4,
    )
    registry = context.freeze_registry()
    last = registry.operations[-1]
    foreign = priors.JointOperationSpan(
        ("graph", "foreign"),
        last.event_stop,
        last.event_stop,
    )
    corrupted = replace(registry, operations=(*registry.operations, foreign))
    with pytest.raises(ValueError, match="typed ownership"):
        plan.validate(corrupted)


def test_graph_plan_bad_geometry_rejects_before_named_rng_changes() -> None:
    rng = np.random.default_rng(43)
    control = np.random.default_rng(43)
    context = priors.JointSamplingContext(rng, mode="local")
    with pytest.raises(ValueError, match="parent"):
        priors.sample_joint_graph_plan(
            context,
            "graph",
            parents=((), (1,)),
            widths=(1, 1),
            calibration_rows=4,
        )
    assert context.freeze_registry().operations == ()
    np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_graph_plan_retains_uniform_weight_owners_with_empty_decision_spans() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(44), mode="local")
    functions = priors.JointFunctionPrior(
        functions="lin",
        matrix=priors.JointMatrixPrior(families=("importance",), vector_family="uniform"),
    )
    plan = priors.sample_joint_graph_plan(
        context,
        "graph",
        parents=((), (0,)),
        widths=(1, 1),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("uniform",), function=functions),
        function_prior=functions,
    )
    registry = context.freeze_registry()
    plan.validate(registry)
    owners = {owner.operation: owner for owner in plan.owners()}
    empty_weights = [
        span for span in registry.operations if owners[span.operation].kind == "weights"
    ]
    assert len(empty_weights) == 2
    assert all(span.event_start == span.event_stop for span in empty_weights)
    missing = replace(
        registry, operations=tuple(span for span in registry.operations if span != empty_weights[0])
    )
    with pytest.raises(ValueError, match="typed ownership"):
        plan.validate(missing)


@pytest.mark.parametrize("field", ["kind", "reference", "operation"])
def test_graph_plan_rejects_mistyped_or_redirected_owner_references(field: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(45), mode="local")
    plan = priors.sample_joint_graph_plan(
        context,
        "graph",
        parents=((), (0,)),
        widths=(1, 1),
        calibration_rows=4,
    )
    registry = context.freeze_registry()
    owners = plan.owners()
    plan.validate(registry, owners=owners)
    if field == "kind":
        changed = replace(owners[2], kind="function")
    elif field == "reference":
        changed = replace(owners[2], reference=("nodes", 1, "mechanism"))
    else:
        changed = replace(owners[2], operation=("graph", "node:1", "mechanism"))
    with pytest.raises(ValueError, match="typed ownership"):
        plan.validate(registry, owners=(*owners[:2], changed, *owners[3:]))
