"""Selected slots bind mixed product source contracts through public sampling."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize("mode", ["local", "global", "meta"])
@pytest.mark.parametrize("catalog", ["prod,kumo-prod", "default,kumo-prod"])
def test_mixed_slots_own_distinct_child_catalogs_and_replay(mode: str, catalog: str) -> None:
    seen: set[bool] = set()
    for seed in range(30):
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode=mode)
        prior = priors.JointFunctionPrior(
            functions=catalog, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
        )
        plan = priors.sample_joint_function(
            context, "f", input_width=2, output_width=1, calibration_rows=8, prior=prior
        )
        if plan.family != "prod":
            continue
        seen.add(plan.kumo_product)
        other_slot = next(
            s
            for s in prior.request().choices
            if prior.family(s) == "prod" and prior.is_kumo_slot(s) != plan.kumo_product
        )
        with pytest.raises(ValueError, match="product child"):
            replace(plan, slot=other_slot)
        assert all(
            c.prior.functions == ("kumo-prod" if plan.kumo_product else "prod")
            for c in plan.children
        )
        assert all(c.family in c.prior.product_children for c in plan.children)
        plan.validate(context.freeze())
        realized = priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=np.random.default_rng(10),
            fit_rng=np.random.default_rng(11),
            calibration=np.random.default_rng(999).normal(size=(8, 2)),
            calibration_id="mixed-fixture",
        )
        assert isinstance(realized, priors.JointProductFunctionRealization)
        assert isinstance(realized.function, priors.FrozenKumoProductFunction) == plan.kumo_product
        assert (
            priors.joint_function_realization_from_dict(
                asdict(realized), decisions=context.freeze()
            )
            == realized
        )
        if seen == {False, True}:
            break
    assert seen == {False, True}


@pytest.mark.parametrize("path", ["explicit", "parent", "native"])
@pytest.mark.parametrize("mode", ["local", "global", "meta"])
def test_mixed_products_travel_through_graph_replay(path: str, mode: str) -> None:
    f = priors.JointFunctionPrior(
        functions="prod,kumo-prod",
        frequencies=4,
        matrix=priors.JointMatrixPrior(families=("gaussian",)),
    )
    root = priors.JointRootPrior(families=("gaussian",), function=f)
    streams = dict(
        zip(
            ("named_rng", "raw_rng", "calibration_rng", "fit_rng"),
            (np.random.default_rng(s) for s in (37, 10, 11, 12)),
            strict=True,
        )
    )
    if path == "native":
        domain = priors.NativeDomainPrior(
            kinds=("categorical",), cardinalities=(3,), emissions=("softmax",)
        )
        request = priors.NativeGraphRequest(
            parents=((), (0,)),
            domains=(domain,) * 2,
            observed=((0, 0),),
            target=(1, 0),
            rows=16,
            context_rows=8,
            calibration_rows=8,
            root_prior=root,
            function_prior=f,
            mode=mode,
        )
        graph = priors.sample_native_graph(request, **streams)
    else:
        task = priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        )
        request = priors.JointGraphRequest(
            task,
            ((), (0,)),
            (2, 2),
            ((0, 0),),
            ((1, 0),),
            root_prior=root,
            function_prior=f,
            calibration_rows=8,
            mode=mode,
        )
        graph = (
            priors.sample_parent_graph(
                request,
                composition=priors.ParentCompositionPrior(
                    source="extension", mode="aggregate", aggregation="sum"
                ),
                **streams,
            )
            if path == "parent"
            else priors.sample_joint_graph(request, **streams)
        )
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    assert priors.generate_task(graph, seed=21, task_id=4).inputs.shape[0] == 16

    before = json.dumps([s.bit_generator.state for s in streams.values()])
    limited = replace(request, limits=priors.CausalLimits(parameters=10))
    if isinstance(limited, priors.NativeGraphRequest):
        with pytest.raises(ValueError, match="limit"):
            priors.sample_native_graph(limited, **streams)
    elif path == "parent":
        with pytest.raises(ValueError, match="limit"):
            priors.sample_parent_graph(
                limited,
                composition=priors.ParentCompositionPrior(
                    source="extension", mode="aggregate", aggregation="sum"
                ),
                **streams,
            )
    else:
        with pytest.raises(ValueError, match="limit"):
            priors.sample_joint_graph(limited, **streams)
    assert json.dumps([s.bit_generator.state for s in streams.values()]) == before
