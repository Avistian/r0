"""Typed scalar laws, joint label siblings and atomic/query contracts through public APIs."""

import json
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def vector_graph() -> p.NativeVectorRecipe:
    continuous = p.VariableDomain()
    binary = p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))
    categorical = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
    root = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (1.0,), (-1.0,), (0.0,), (1.0,))),
            p.JointRootBase("gaussian", 1),
        ),
        (continuous, binary, categorical),
    )
    child = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(
                ((1.0, 0.0, 10.0, 0.0, 1.0, 2.0), (1.0, 0.0, 10.0, 0.0, 1.0, 2.0))
            ),
        ),
        (continuous, binary),
    )
    return p.NativeVectorRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=4, signal_features=4, active_features=4),
        (root, child),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        (1, 0),
    )


def sample(request: p.TypedInterventionPolicyRequest):
    return p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
@pytest.mark.parametrize("emission", ["softmax", "argmax", "threshold"])
def test_native_scalar_category_policy_literal_law(
    engine: Any, emission: Literal["softmax", "argmax", "threshold"]
):
    domain = p.FrequencyVariableDomain(
        "categorical",
        ("a", "b", "c"),
        emission,
        (-1.0, 1.0) if emission == "threshold" else (),
        frequencies=(0.2, 0.3, 0.5),
    )
    matrix = ((1.0,),) if emission == "threshold" else ((-1.0,), (0.0,), (1.0,))
    base = p.NativeCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.NativeCausalNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(matrix), p.JointRootBase("gaussian", 1)
                ),
                domain,
            ),
            p.NativeCausalNode(
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 5.0, 11.0),)))
            ),
        ),
        ((0, 0),),
        (1, 0),
    )
    address = p.InterventionPolicyLocation(
        0, "native-variable", variable=0, location="native-value", probabilities=(0.6, 0.3, 0.1)
    )
    request = p.TypedInterventionPolicyRequest(
        base, (1,), (address,), eligibility=(0,), count=1, mode="matched"
    )
    receipt = p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.Generator(engine(3)),
        value_rng=np.random.Generator(engine(5)),
        calibration_rng=np.random.Generator(engine(7)),
    )
    chosen = int(
        np.searchsorted(
            np.cumsum([0.6, 0.3, 0.1]), np.random.Generator(engine(5)).random(), side="right"
        )
    )
    pair = p.generate_typed_policy_pair(receipt, seed=11, task_id=13)
    np.testing.assert_array_equal(pair.after.inputs, chosen)
    np.testing.assert_allclose(pair.after.targets, [2.0, 5.0, 11.0][chosen])
    assert isinstance(receipt.request.base, p.NativeCausalRecipe)
    assert receipt.request.base.nodes[0].domain == domain
    assert not receipt.reference_raw


def test_typed_hard_sibling_allowed_but_label_address_and_soft_joint_label_forbidden():
    from dataclasses import replace

    base = vector_graph()
    address = p.InterventionPolicyLocation(
        1, "native-variable", variable=1, location="native-value", probabilities=(0.0, 1.0)
    )
    request = p.TypedInterventionPolicyRequest(
        base, (1,), (address,), eligibility=(1,), count=1, mode="matched"
    )
    pair = p.generate_typed_policy_pair(sample(request), seed=17, task_id=5)
    assert torch.equal(pair.after.targets, pair.before.targets)
    np.testing.assert_array_equal(pair.after.inputs[:, -1], 1)
    with pytest.raises(ValueError, match="target variable"):
        replace(request, locations=(replace(address, variable=0, probabilities=()),))
    with pytest.raises(ValueError, match="label equations"):
        replace(
            request,
            kind="soft",
            locations=(),
            parameter_laws=(p.InterventionParameterLaw(1, ("matrix",)),),
        )


def test_typed_query_invariance_and_failed_parameter_draw_rollback():
    import copy
    from dataclasses import replace

    base = vector_graph()
    address = p.InterventionPolicyLocation(
        0, "native-variable", variable=0, location="native-value"
    )
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (1,), (address,), eligibility=(0,), count=1, reference_rows=32
        )
    )
    changed = receipt.with_task(replace(receipt.task, rows=40, context_rows=17))
    assert changed.bounds == receipt.bounds
    assert changed.reference_raw == receipt.reference_raw
    assert changed.runtime.interventions == receipt.runtime.interventions
    assert p.recipe_from_dict(asdict(changed)) == changed
    request = p.TypedInterventionPolicyRequest(
        base,
        (1,),
        kind="soft",
        eligibility=(0,),
        count=1,
        parameter_laws=(p.InterventionParameterLaw(0, ("matrix",), location=1e308, scale=1e308),),
    )
    streams = [np.random.default_rng(i) for i in (3, 13, 7)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite"):
        p.sample_typed_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
    assert [r.bit_generator.state for r in streams] == before


def test_typed_outer_cap_preserves_pair_base_identity():
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, active_features=1, signal_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    address = p.InterventionPolicyLocation(0, channels=(0,))
    request = p.TypedInterventionPolicyRequest(
        base,
        (1,),
        (address,),
        eligibility=(0,),
        count=1,
        limits=p.CausalLimits(working_bytes=128 * 1024**2),
    )
    receipt = sample(request)
    pair = p.generate_typed_policy_pair(receipt, seed=11, task_id=13)
    assert json.loads(pair.before.recipe_json)["recipe"] == json.loads(json.dumps(asdict(base)))
    p.compare_causal_pair(
        pair.before, pair.after, limits=p.CausalLimits(working_bytes=64 * 1024**2)
    )
