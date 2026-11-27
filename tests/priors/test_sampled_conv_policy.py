"""Soft interventions retain sampled kernels while authoring only replacement parameters."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors as p


def graph(kind: str, wrapper: str):
    sampled = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(
            3,
            1,
            (
                p.Conv1DLayerPrior(
                    (p.Conv1DGeometry(1, 3),),
                    activations=(p.CausalActivation(),),
                ),
            ),
        ),
        named_rng=np.random.default_rng(51),
        raw_rng=np.random.default_rng(52),
    )
    function = sampled
    prefix = ()
    if wrapper == "child":
        function = p.FrozenParentFunction((sampled,))
        prefix = ("children", "0")
    elif wrapper == "joint":
        function = p.FrozenParentFunction(
            (p.FrozenLinearFunction(((1.0, 2.0, 3.0),)),), joint=sampled
        )
        prefix = ("joint",)
    categorical = kind in ("native", "vector")
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(
            ((1.0,), (-1.0,), (0.0,))
            if categorical
            else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
        p.JointRootBase("gaussian", 1 if categorical else 3),
    )
    nodes = (
        root,
        p.FunctionalCausalNode((0,), function),
        p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((2.0,),))),
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=5,
        features=1 if categorical else 3,
        active_features=1 if categorical else 3,
        signal_features=1 if categorical else 3,
    )
    if categorical:
        domains = (
            p.VariableDomain("categorical", ("a", "b", "c"), "argmax"),
            p.VariableDomain(),
            p.VariableDomain(),
        )
        if kind == "native":
            base = p.NativeCausalRecipe(
                task,
                tuple(p.NativeCausalNode(n, d) for n, d in zip(nodes, domains, strict=True)),
                ((0, 0),),
                (2, 0),
            )
        else:
            base = p.NativeVectorRecipe(
                task,
                tuple(p.NativeVectorNode(n, (d,)) for n, d in zip(nodes, domains, strict=True)),
                ((0, 0),),
                (2, 0),
            )
    elif kind == "converted":
        base = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            tuple(p.ConvertedReadout(0, channel=i) for i in range(3)),
            p.ConvertedReadout(2, channel=0),
        )
    else:
        base = p.FunctionalCausalRecipe(task, nodes, ((0, 0), (0, 1), (0, 2)), ((2, 0),))
    return base, sampled, prefix + ("function", "layers", "0", "kernel")


@pytest.mark.parametrize("kind", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("wrapper", ["direct", "child", "joint"])
@pytest.mark.parametrize("family", ["normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0"])
def test_sampled_conv_policy_literal_values(kind: str, wrapper: str, family: str):
    base, original, path = graph(kind, wrapper)
    law = p.InterventionParameterLaw(
        1, path, family, 0.5, 2.0, (1.0, -2.0) if family == "choice-r0" else ()
    )
    request = p.TypedInterventionPolicyRequest(
        base, (2,), kind="soft", mode="matched", eligibility=(1,), count=1, parameter_laws=(law,)
    )
    policy = p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    assert policy.request.base == base
    change = policy.runtime.interventions[0]
    assert isinstance(change, p.SoftIntervention)
    replacement = change.function
    if wrapper == "direct":
        conv = replacement
    else:
        assert isinstance(replacement, p.FrozenParentFunction)
        conv = replacement.children[0] if wrapper == "child" else replacement.joint
    assert type(conv) is p.FrozenFeatureConv1D
    raw = next(draw.array() for draw in policy.raw if draw.role == "parameter:0")
    expected = (
        0.5 + 2 * raw
        if family in ("normal-r0", "uniform-r0")
        else np.exp((1 - raw) * np.log(0.5) + raw * np.log(2.0))
        if family == "log-uniform-r0"
        else np.array([1.0, -2.0])[np.floor(raw * 2).astype(int)]
    )
    np.testing.assert_array_equal(np.array(conv.layers[0].kernel).ravel(), expected)
    pair = p.generate_typed_policy_pair(policy, seed=11, task_id=13)
    before, after = pair.before, pair.after
    inputs = (
        np.eye(3)[before.inputs.numpy()[:, 0].astype(int)]
        if kind in ("native", "vector")
        else before.inputs.numpy()
    )
    values = inputs @ expected + original.function.layers[0].bias[0]
    if wrapper == "joint":
        values += inputs @ np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(after.scores.numpy()[:, 0], 2 * values, atol=1e-12)
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(policy))))
    assert restored == policy


@pytest.mark.parametrize("family", ["normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0"])
def test_scalar_policy_uses_authored_replacement_and_separate_draws(family: str):
    base, original, path = graph("functional", "joint")
    assert isinstance(base, p.FunctionalCausalRecipe)
    request = p.InterventionPolicyRequest(
        base,
        (2,),
        kind="soft",
        profile="explicit-policy-r0-v1",
        mode="matched",
        eligibility=(1,),
        count=1,
        parameter_laws=(
            p.InterventionParameterLaw(
                1,
                path,
                family,
                0.5,
                2.0,
                (1.0, -2.0) if family == "choice-r0" else (),
            ),
        ),
    )
    sampled = p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    assert sampled.request.base == base
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(sampled)))) == sampled
    pair = p.generate_policy_pair(sampled, seed=11, task_id=13)
    trace = p.explain_generation(pair.after, [2, 0, 2])
    assert any(n["id"].startswith("policy-original-1-0-construction-draw") for n in trace["nodes"])
    assert any(n["id"].startswith("policy-replacement-1-") for n in trace["nodes"])
    change = sampled.runtime.interventions[0]
    assert isinstance(change, p.SoftIntervention)
    assert isinstance(change.function, p.FrozenParentFunction)
    assert isinstance(change.function.joint, p.FrozenFeatureConv1D)
    values = pair.before.inputs.numpy()
    expected = 2 * (
        values
        @ (np.array(change.function.joint.layers[0].kernel).ravel() + np.array([1.0, 2.0, 3.0]))
        + original.function.layers[0].bias[0]
    )
    np.testing.assert_allclose(pair.after.scores.numpy()[:, 0], expected, atol=1e-12)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "path",
    [
        ("request", "layers", "0", "kernel_scale"),
        ("raw", "0", "values"),
        ("initial", "0", "state"),
    ],
)
def test_policy_cannot_resample_initializer_custody(typed: bool, path: tuple[str, ...]):
    base, _, _ = graph("functional", "direct")
    assert isinstance(base, p.FunctionalCausalRecipe)
    law = p.InterventionParameterLaw(1, path)
    constructor = p.TypedInterventionPolicyRequest if typed else p.InterventionPolicyRequest
    with pytest.raises(ValueError, match="custody|parameter path"):
        constructor(
            base,
            (2,),
            kind="soft",
            eligibility=(1,),
            parameter_laws=(law,),
            profile="explicit-typed-policy-r0-v1" if typed else "explicit-policy-r0-v1",
        )


def test_unselected_sampled_joint_stays_replay_bound_and_trace_custody():
    from dataclasses import replace

    base, original, path = graph("functional", "child")
    assert isinstance(base, p.FunctionalCausalRecipe)
    parent = p.FrozenParentFunction((original,), joint=original)
    base = replace(
        base, nodes=(base.nodes[0], replace(base.nodes[1], function=parent), base.nodes[2])
    )
    request = p.TypedInterventionPolicyRequest(
        base,
        (2,),
        kind="soft",
        mode="matched",
        eligibility=(1,),
        parameter_laws=(p.InterventionParameterLaw(1, path),),
    )
    sampled = p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    change = sampled.runtime.interventions[0]
    assert isinstance(change, p.SoftIntervention)
    assert isinstance(change.function, p.FrozenParentFunction)
    assert isinstance(change.function.children[0], p.FrozenFeatureConv1D)
    assert change.function.joint == original
    pair = p.generate_typed_policy_pair(sampled, seed=11, task_id=13)
    trace = p.explain_generation(pair.after, [2, 0, 2])
    assert any(n["id"].startswith("policy-original-1-0-construction-draw") for n in trace["nodes"])
    realization = next(n for n in trace["nodes"] if n["id"] == "policy-realization")
    assert realization["parameters"]["original_sampled_construction"]


@pytest.mark.parametrize("typed", [False, True])
def test_sampled_conv_overflow_rolls_back_all_streams(typed: bool):
    base, _, path = graph("functional", "direct")
    assert isinstance(base, p.FunctionalCausalRecipe)
    law = p.InterventionParameterLaw(1, path, location=1e308, scale=1e308)
    request = (
        p.TypedInterventionPolicyRequest(
            base, (2,), kind="soft", eligibility=(1,), parameter_laws=(law,)
        )
        if typed
        else p.InterventionPolicyRequest(
            base,
            (2,),
            kind="soft",
            profile="explicit-policy-r0-v1",
            eligibility=(1,),
            parameter_laws=(law,),
        )
    )
    streams = [np.random.default_rng(i) for i in (3, 4, 7)]
    states = [json.dumps(rng.bit_generator.state, sort_keys=True) for rng in streams]

    def attempt():
        if isinstance(request, p.TypedInterventionPolicyRequest):
            return p.sample_typed_intervention_policy(
                request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
            )
        return p.sample_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )

    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite resampled"):
        attempt()
    assert states == [json.dumps(rng.bit_generator.state, sort_keys=True) for rng in streams]
