"""Public typed policy probes use literal category/state arithmetic and retained archives."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


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


def converted_graph(mode: str = "neighbor_disc", *, noise: float = 0.2) -> p.ConvertedCausalRecipe:
    converter = p.CategoricalConverter(
        "kind",
        0,
        2,
        ("left", "right"),
        mode,
        ((-2.0, -3.0), (4.0, 5.0)),
        mean=(0.0, 0.0) if mode.startswith("softmax") else (),
        scale=(1.0, 1.0) if mode.startswith("softmax") else (),
        bias=(0.0, 0.0) if mode.startswith("softmax") else (),
        post_function=p.FrozenLinearFunction(((2.0, 0.0), (0.0, 3.0)))
        if mode == "neighbor_func"
        else None,
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (-1.0,))),
            p.JointRootBase("gaussian", 1),
            noise_scale=(noise, noise),
            pipeline=p.CausalNodePipeline(2, node_importance=2.0),
        ),
        (converter,),
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(((2.0, 1.0),)),
        )
    )
    return p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (root, child),
        (p.ConvertedReadout(0, capture="kind"),),
        p.ConvertedReadout(1, channel=0),
    )


def sample(request: p.TypedInterventionPolicyRequest):
    return p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )


def test_vector_categories_preserve_joint_siblings_and_full_receipt(tmp_path: Path):
    base = vector_graph()
    location = p.InterventionPolicyLocation(
        0, "native-variable", variable=2, location="native-value", probabilities=(0.1, 0.2, 0.7)
    )
    request = p.TypedInterventionPolicyRequest(
        base, (1,), (location,), eligibility=(0,), count=1, mode="matched"
    )
    receipt = sample(request)
    assert receipt.runtime.base == base
    pair = p.generate_typed_policy_pair(receipt, seed=17, task_id=2**55 + 3)
    u = np.random.default_rng(5).random()
    category = int(np.searchsorted(np.cumsum([0.1, 0.2, 0.7]), u, side="right"))
    np.testing.assert_array_equal(pair.after.inputs[:, :2], pair.before.inputs[:, :2])
    np.testing.assert_array_equal(pair.after.inputs[:, 2], category)
    np.testing.assert_allclose(
        pair.after.targets, pair.after.inputs[:, 0] + 10 * pair.after.inputs[:, 1] + category
    )
    assert not receipt.reference_raw
    assert p.recipe_from_dict(asdict(receipt)) == receipt
    trace = p.explain_generation(pair.after, [31, 0, 31])
    assert trace["rows"] == [31, 0, 31]
    assert any(n["id"] == "policy-selection" for n in trace["nodes"])
    realization = next(n for n in trace["nodes"] if n["id"] == "policy-realization")
    assert "policy-draw-1" in realization["parents"]
    quantile = next(n for n in trace["nodes"] if n["id"] == "policy-draw-1")
    np.testing.assert_allclose(quantile["values"], [[u], [u], [u]])
    path = tmp_path / "pair"
    save_causal_pair(path, pair, max_bytes=5_000_000)
    assert load_causal_pair(path).after.recipe_json == pair.after.recipe_json


@pytest.mark.parametrize(
    "mode",
    [
        "neighbor_id",
        "neighbor_disc",
        "neighbor_func",
        "neighbor_int",
        "softmax_id",
        "softmax_disc",
        "softmax_int",
    ],
)
def test_converted_capture_policy_preserves_descendants(mode: str):
    base = converted_graph(mode)
    address = p.InterventionPolicyLocation(
        0, "capture", capture="kind", location="capture", probabilities=(0.0, 1.0)
    )
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (1,), (address,), eligibility=(0,), count=1, mode="matched"
        )
    )
    pair = p.generate_typed_policy_pair(receipt, seed=23, task_id=5)
    assert torch.equal(pair.before.targets, pair.after.targets)
    assert torch.equal(pair.after.inputs, torch.ones_like(pair.after.inputs))
    trace = p.explain_generation(pair.after, [15, 0, 15])
    assert trace["sampled_policy"]["locations"][0]["location"] == "capture"
    assert json.loads(pair.after.recipe_json)["recipe"]["request"]["base"] == asdict_json(base)


def asdict_json(value: Any):
    return json.loads(json.dumps(asdict(value)))


@pytest.mark.parametrize("lifecycle", ["independent-calibration-r0", "source-reference-population"])
def test_vector_continuous_reference_literal_rng_and_bounds(lifecycle: str):
    import hashlib

    base = vector_graph()
    address = p.InterventionPolicyLocation(
        0, "native-variable", variable=0, location="native-value"
    )
    request = p.TypedInterventionPolicyRequest(
        base,
        (1,),
        (address,),
        eligibility=(0,),
        count=1,
        mode="matched",
        reference_rows=32,
        bounds_lifecycle=lifecycle,
        reference_seed=101,
        reference_task_id=103,
    )
    receipt = sample(request)
    seed = (
        int(np.random.default_rng(7).integers(2**63 - 1))
        if lifecycle.startswith("independent")
        else 101
    )
    prefix = (
        ("typed-policy-independent-reference-v1",) if lifecycle.startswith("independent") else ()
    )
    key = json.dumps(
        ("native-vector-causal-v1", seed, 103, (*prefix, "root", 0)),
        sort_keys=True,
        separators=(",", ":"),
    )
    reference = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16)).normal(
        size=32
    )
    u = np.random.default_rng(5).random()
    expected = (1 - u) * reference.min() + u * reference.max()
    np.testing.assert_allclose(receipt.bounds[0].minimum, [reference.min()])
    np.testing.assert_allclose(receipt.bounds[0].maximum, [reference.max()])
    pair = p.generate_typed_policy_pair(receipt, seed=17, task_id=5)
    np.testing.assert_allclose(pair.after.inputs[:, 0], expected)
    np.testing.assert_array_equal(pair.after.inputs[:, 1:3], pair.before.inputs[:, 1:3])
    assert receipt.reference_raw
    assert p.recipe_from_dict(asdict(receipt)) == receipt


def test_sampled_native_base_custody_survives_policy_and_tampering():
    import copy

    prior = p.NativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3,),
        emissions=("softmax",),
        frequency_laws=("dirichlet",),
    )
    request = p.NativeGraphRequest(
        parents=((), (0,)),
        domains=(prior, prior),
        observed=((0, 0),),
        target=(1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=16,
        root_prior=p.JointRootPrior(
            families=("gaussian",), function=p.JointFunctionPrior(functions="lin")
        ),
        function_prior=p.JointFunctionPrior(functions="lin"),
    )
    base = p.sample_native_graph(
        request,
        named_rng=np.random.default_rng(1),
        raw_rng=np.random.default_rng(2),
        calibration_rng=np.random.default_rng(3),
        fit_rng=np.random.default_rng(4),
    )
    location = p.InterventionPolicyLocation(
        0, "native-variable", variable=0, location="native-value"
    )
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (1,), (location,), eligibility=(0,), count=1, reference_rows=16
        )
    )
    assert asdict(receipt.request.base) == asdict(base)
    pair = p.generate_typed_policy_pair(receipt, seed=17, task_id=5)
    assert json.loads(pair.before.recipe_json)["recipe"] == asdict_json(base)
    assert p.recipe_from_dict(asdict(receipt)) == receipt
    altered = copy.deepcopy(asdict(receipt))
    altered["request"]["base"]["version"] = "forged"
    with pytest.raises(ValueError, match="version|unexpected|unsupported|invalid|Unknown|unknown"):
        p.recipe_from_dict(altered)


@pytest.mark.parametrize(
    "mode", ["neighbor_disc", "neighbor_func", "neighbor_int", "softmax_disc", "softmax_int"]
)
def test_converted_representation_policy_literal_descendant(mode: str):
    base = converted_graph(mode, noise=0.0)
    address = p.InterventionPolicyLocation(
        0,
        "capture",
        capture="kind",
        location="representation_and_capture",
        probabilities=(0.0, 1.0),
    )
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (1,), (address,), eligibility=(0,), count=1, mode="matched"
        )
    )
    pair = p.generate_typed_policy_pair(receipt, seed=23, task_id=5)
    expected = 62.0 if mode == "neighbor_func" else 6.0 if mode.endswith("_int") else 26.0
    np.testing.assert_allclose(pair.after.targets, expected)
    assert p.recipe_from_dict(asdict(receipt)) == receipt


@pytest.mark.parametrize("family", ["normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0"])
def test_typed_soft_equations_keep_domains_and_declared_law(family: str):
    base = vector_graph()
    law = p.InterventionParameterLaw(
        0, ("matrix",), family, 0.2, 2.0, (-2.0, 4.0) if family == "choice-r0" else ()
    )
    request = p.TypedInterventionPolicyRequest(
        base, (1,), kind="soft", eligibility=(0,), count=1, parameter_laws=(law,), mode="matched"
    )
    receipt = sample(request)
    rng = np.random.default_rng(5)
    u = rng.normal(size=5) if family == "normal-r0" else rng.random(5)
    expected = (
        0.2 + 2 * u
        if family in ("normal-r0", "uniform-r0")
        else np.exp((1 - u) * np.log(0.2) + u * np.log(2.0))
        if family == "log-uniform-r0"
        else np.asarray([-2.0, 4.0])[np.floor(u * 2).astype(int)]
    )
    change = receipt.runtime.interventions[0]
    assert isinstance(change, p.SoftIntervention)
    assert isinstance(change.function, p.FrozenLinearFunction)
    np.testing.assert_allclose(np.asarray(change.function.matrix).ravel(), expected)
    pair = p.generate_typed_policy_pair(receipt, seed=17, task_id=5)
    np.testing.assert_allclose(
        pair.after.targets,
        pair.after.inputs[:, 0] + 10 * pair.after.inputs[:, 1] + pair.after.inputs[:, 2],
    )
    assert asdict(receipt.request.base) == asdict(base)
    assert not receipt.reference_raw


def test_typed_pre_stream_admission_and_strict_mutations():
    import copy
    from dataclasses import replace
    from unittest.mock import patch

    base = vector_graph()
    address = p.InterventionPolicyLocation(
        0, "native-variable", variable=0, location="native-value"
    )
    request = p.TypedInterventionPolicyRequest(
        base, (1,), (address,), eligibility=(0,), count=1, reference_rows=32
    )
    receipt = sample(request)
    with (
        patch.object(p.RawDrawLedger, "transactions", side_effect=AssertionError("stream touched")),
        pytest.raises(ValueError, match="limit"),
    ):
        sample(replace(request, limits=p.CausalLimits(working_bytes=1)))
    altered = copy.deepcopy(asdict(receipt))
    altered["request"]["reference_rows"] = 20000
    altered["request"]["limits"]["working_bytes"] = 512 * 1024
    with (
        patch.object(
            type(receipt.initial[0]), "restore", side_effect=AssertionError("snapshot restored")
        ),
        pytest.raises(ValueError, match="limit"),
    ):
        p.recipe_from_dict(altered)
    for field in ("selected", "bounds", "initial", "final", "raw", "reference_raw", "runtime"):
        malformed = copy.deepcopy(asdict(receipt))
        del malformed[field]
        with pytest.raises(ValueError, match="requires every field"):
            p.recipe_from_dict(malformed)
    pair = p.generate_typed_policy_pair(receipt, seed=17, task_id=5)
    with pytest.raises(ValueError, match="semantic"):
        p.explain_generation(replace(pair.after, task_type="binary"), [0])


def test_fixed_fitted_envelope_retains_original_fit_without_backend_refit(
    monkeypatch: pytest.MonkeyPatch,
):
    tree = pytest.importorskip("sklearn.tree")

    skeleton = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
        ),
        ((0, 0),),
        ((2, 0),),
    )
    base = p.sample_fitted_graph(
        skeleton,
        ((1, p.FittedTreePlan()),),
        prefix_rng=np.random.default_rng(1),
        input_rng=np.random.default_rng(2),
        target_rng=np.random.default_rng(3),
        backend_rng=np.random.default_rng(4),
    )

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("backend refit")

    monkeypatch.setattr(tree.DecisionTreeRegressor, "fit", forbidden)
    address = p.InterventionPolicyLocation(1, channels=(0,))
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (2,), (address,), eligibility=(1,), count=1, reference_rows=16
        )
    )
    assert asdict(receipt.request.base) == asdict(base)
    pair = p.generate_typed_policy_pair(receipt, seed=11, task_id=13)
    assert json.loads(pair.before.recipe_json)["recipe"] == asdict_json(base)
    change = receipt.runtime.interventions[0]
    assert isinstance(change, p.HardIntervention)
    np.testing.assert_allclose(pair.after.targets, 3 * change.values[0])
    assert p.recipe_from_dict(asdict(receipt)) == receipt


def sampled_vector_request(
    parent: bool = False, function: str = "lin"
) -> p.NativeVectorGraphRequest:
    c = p.NativeDomainPrior(kinds=("continuous",))
    b = p.NativeDomainPrior(
        kinds=("binary",),
        emissions=("threshold",),
        frequency_laws=("uniform",),
        threshold_laws=("uniform-quantile",),
    )
    k = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 4.0),),
    )
    k5 = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(5,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 3.0, 4.0, 5.0),),
    )
    f = p.JointFunctionPrior(functions=function, matrix=p.JointMatrixPrior(families=("gaussian",)))
    return p.NativeVectorGraphRequest(
        parents=((), (0,)),
        domains=((c, b, k), (k5, c, b)),
        observed=((0, 0), (0, 1), (0, 2), (1, 1)),
        target=(1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="sum"
        )
        if parent
        else None,
    )


def test_sampled_vector_policy_retains_all_four_construction_streams():
    base = p.sample_native_vector_graph(
        sampled_vector_request(),
        **{
            name: np.random.default_rng(i)
            for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )
    address = p.InterventionPolicyLocation(
        0, "native-variable", variable=2, location="native-value"
    )
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base, (1,), (address,), eligibility=(0,), count=1, reference_rows=16
        )
    )
    assert asdict(receipt.request.base) == asdict(base)
    assert p.recipe_from_dict(asdict(receipt)) == receipt
    pair = p.generate_typed_policy_pair(receipt, seed=11, task_id=13)
    assert json.loads(pair.before.recipe_json)["recipe"] == asdict_json(base)


def test_sampled_converted_policy_retains_fit_raw_and_capture():
    functions = p.JointFunctionPrior(
        functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",))
    )
    request = p.ConvertedGraphRequest(
        p.AttachmentGraphPrior(nodes=2),
        (("a", "b", "c"), ()),
        ("no", "yes"),
        rows=16,
        context_rows=8,
        calibration_rows=16,
        width_prior=p.JointWidthPrior(low=1, high=2),
        function_prior=functions,
        root_prior=p.JointRootPrior(families=("gaussian",), function=functions),
        converter_prior=p.CategoricalConverterPrior(
            modes=("softmax_disc",),
            function=functions,
            embeddings=p.JointRootPrior(families=("gaussian",), function=functions),
        ),
        pipeline_prior=p.JointPipelinePrior(
            channel_importance=False,
            l2_normalization=False,
            converter=p.JointNumericalConverterPrior(
                allow_kumaraswamy=False, allow_activation=False
            ),
        ),
    )
    base = p.sample_converted_graph(
        request,
        **{
            name: np.random.default_rng(i)
            for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )
    candidates = [
        (i, c.name)
        for i, n in enumerate(base.recipe.nodes)
        for c in n.converters
        if not (i == base.recipe.target.node and c.name == base.recipe.target.capture)
    ]
    node, name = candidates[0]
    address = p.InterventionPolicyLocation(node, "capture", capture=name, location="capture")
    receipt = sample(
        p.TypedInterventionPolicyRequest(
            base,
            (base.recipe.target.node,),
            (address,),
            eligibility=(node,),
            count=1,
            reference_rows=16,
            mode="matched",
        )
    )
    assert asdict(receipt.request.base) == asdict(base)
    assert p.recipe_from_dict(asdict(receipt)) == receipt
    pair = p.generate_typed_policy_pair(receipt, seed=11, task_id=13)
    assert torch.equal(pair.before.targets, pair.after.targets)
