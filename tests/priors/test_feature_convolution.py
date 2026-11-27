"""Feature-axis kernels are tied across positions and never across dataset rows."""

import copy
import json
from dataclasses import asdict, replace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import priors as p


def literal(function: p.FrozenFeatureConv1D, values: NDArray[np.float64]):
    x = (
        values.reshape(len(values), function.input_channels, function.input_length)
        if function.layout == "channel-major"
        else values.reshape(len(values), function.input_length, function.input_channels).transpose(
            0, 2, 1
        )
    )
    for layer in function.layers:
        g = layer.geometry
        effective = g.dilation * (g.kernel_size - 1) + 1
        left, right = g.explicit_padding if g.padding == "explicit" else (0, 0)
        if g.padding == "same-upper":
            n = (x.shape[2] + g.stride - 1) // g.stride
            total = max(0, (n - 1) * g.stride + effective - x.shape[2])
            left, right = total // 2, total - total // 2
        length = (x.shape[2] + left + right - effective) // g.stride + 1
        kernel = np.asarray(layer.kernel)
        out = np.zeros((len(x), g.output_channels, length))
        for row in range(len(x)):
            for oc in range(g.output_channels):
                group = oc // (g.output_channels // g.groups)
                for pos in range(length):
                    out[row, oc, pos] = layer.bias[oc]
                    for local in range(x.shape[1] // g.groups):
                        for tap in range(g.kernel_size):
                            source = pos * g.stride - left + tap * g.dilation
                            if 0 <= source < x.shape[2]:
                                out[row, oc, pos] += (
                                    kernel[oc, local, tap]
                                    * x[row, group * (x.shape[1] // g.groups) + local, source]
                                )
        if g.residual:
            out += x
        x = out
    return (x if function.layout == "channel-major" else x.transpose(0, 2, 1)).reshape(len(x), -1)


@pytest.mark.parametrize("layout", ["channel-major", "position-major"])
@pytest.mark.parametrize(
    ("padding", "stride", "dilation", "kernel_size", "groups"),
    [
        ("valid", 1, 1, 3, 1),
        ("same-upper", 1, 1, 2, 1),
        ("same-upper", 2, 2, 4, 2),
        ("explicit", 2, 1, 2, 2),
        ("valid", 1, 2, 2, 1),
    ],
)
def test_literal_grouped_dilated_convolution(
    layout: str, padding: str, stride: int, dilation: int, kernel_size: int, groups: int
):
    g = p.Conv1DGeometry(
        4,
        kernel_size,
        stride,
        dilation,
        padding,
        groups,
        (2, 1) if padding == "explicit" else (0, 0),
    )
    kernels = (
        np.arange(4 * (2 // groups) * kernel_size, dtype=np.float64).reshape(
            4, 2 // groups, kernel_size
        )
        / 7
        - 0.5
    )
    layer = p.FeatureConv1DLayer(
        g, tuple(tuple(tuple(map(float, t)) for t in o) for o in kernels), (0.0, 0.2, -0.1, 0.4)
    )
    function = p.FrozenFeatureConv1D(7, 2, (layer,), layout)
    x = np.arange(42, dtype=np.float64).reshape(3, 14) / 11 - 0.7
    actual = p.evaluate_feature_conv1d(function, torch.tensor(x)).numpy()
    np.testing.assert_allclose(actual, literal(function, x), atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(
        p.evaluate_feature_conv1d(function, torch.tensor(x[[2, 0, 2]])).numpy(), actual[[2, 0, 2]]
    )
    assert p.feature_conv1d_from_dict(json.loads(json.dumps(asdict(function)))) == function


def test_impulse_translation_shared_kernel_and_feature_permutation():
    layer = p.FeatureConv1DLayer(
        p.Conv1DGeometry(1, 3, padding="same-upper"), (((2.0, 3.0, 5.0),),), (0.0,)
    )
    function = p.FrozenFeatureConv1D(9, 1, (layer,))
    x = torch.zeros((2, 9), dtype=torch.float64)
    x[0, 3] = 1
    x[1, 4] = 1
    actual = p.evaluate_feature_conv1d(function, x)
    assert torch.equal(
        actual[0], torch.tensor([0.0, 0.0, 5.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert torch.equal(actual[0, :-1], actual[1, 1:])
    perm = torch.tensor([8, 1, 3, 4, 5, 6, 7, 2, 0])
    assert not torch.equal(p.evaluate_feature_conv1d(function, x[:, perm]), actual[:, perm])
    assert layer.parameter_count == 4


@pytest.mark.parametrize("scope", ["flat", "channels", "positions"])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_activation_axes_and_placement(scope: str, placement: str):
    geometry = p.Conv1DGeometry(2, 1)
    layer = p.FeatureConv1DLayer(
        geometry,
        (((2.0,), (0.0,)), ((0.0,), (-1.0,))),
        (0.1, -0.2),
        p.CausalActivation("softmax"),
        placement,
        scope,
    )
    function = p.FrozenFeatureConv1D(3, 2, (layer,))
    x = np.array([[0.0, 1.0, 3.0, 2.0, -1.0, 0.5]])
    shaped = x.reshape(1, 2, 3)

    def activate(v: NDArray[np.float64]):
        shape = v.shape
        axis = 2 if scope == "positions" else 1
        if scope == "flat":
            v = v.reshape(1, -1)
            axis = 1
        exponent = np.exp(v - v.max(axis=axis, keepdims=True))
        return (exponent / exponent.sum(axis=axis, keepdims=True)).reshape(shape)

    transformed = activate(shaped) if placement == "before" else shaped
    expected = (
        transformed * np.array([2.0, -1.0])[None, :, None] + np.array([0.1, -0.2])[None, :, None]
    )
    if placement == "after":
        expected = activate(expected)
    np.testing.assert_allclose(
        p.evaluate_feature_conv1d(function, torch.tensor(x)).numpy(),
        expected.reshape(1, -1),
        atol=1e-15,
    )


def test_stack_residual_and_nonlinear_finite_failure():
    layer = p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1, residual=True), (((2.0,),),), (1.0,))
    function = p.FrozenFeatureConv1D(4, 1, (layer, layer))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    assert torch.equal(p.evaluate_feature_conv1d(function, x), 9 * x + 4)
    bad = replace(layer, kernel=(((1e308,),),), activation=p.CausalActivation("sigmoid"))
    with pytest.raises(ValueError, match="nonfinite"):
        p.evaluate_feature_conv1d(replace(function, layers=(bad,)), x)


def request(law: str = "normal"):
    prior = p.Conv1DLayerPrior(
        (p.Conv1DGeometry(2, 3, padding="same-upper"),),
        kernel_laws=(law,),
        activations=(p.CausalActivation(),),
        kernel_scale=0.7,
        kernel_mean=0.2,
        bias_scale=0.3,
        bias_mean=-0.1,
    )
    return p.FeatureConv1DRequest(5, 2, (prior,))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("law", ["normal", "uniform", "student_t", "cauchy", "rademacher"])
def test_sampling_raw_formula_exact_archive_and_replay(engine: str, law: str):
    named = np.random.Generator(getattr(np.random, engine)(3))
    raw = np.random.Generator(getattr(np.random, engine)(7))
    sample = p.sample_feature_conv1d(request(law), named_rng=named, raw_rng=raw)
    sample.validate_replay()
    assert len(sample.raw) == 5
    kernel = sample.raw[3].array()
    bias = sample.raw[4].array()
    oracle = np.random.Generator(getattr(np.random, engine)(7))
    if law == "normal":
        raw_expected = oracle.normal(size=(2, 2, 3))
    elif law == "uniform":
        raw_expected = oracle.uniform(-1, 1, size=(2, 2, 3))
    elif law == "student_t":
        raw_expected = oracle.standard_t(3, size=(2, 2, 3))
    elif law == "cauchy":
        raw_expected = oracle.standard_cauchy(size=(2, 2, 3))
    else:
        raw_expected = oracle.integers(2, size=(2, 2, 3))
    np.testing.assert_array_equal(kernel, raw_expected)
    np.testing.assert_array_equal(bias, oracle.normal(size=2))
    if law == "rademacher":
        kernel = 2 * kernel - 1
    expected = 0.2 + kernel * (0.7 / np.sqrt(6))
    np.testing.assert_array_equal(np.asarray(sample.function.layers[0].kernel), expected)
    np.testing.assert_array_equal(np.asarray(sample.function.layers[0].bias), -0.1 + 0.3 * bias)
    decoded = p.feature_conv1d_from_dict(json.loads(json.dumps(asdict(sample))))
    assert decoded == sample
    x = torch.ones((2, 10), dtype=torch.float64)
    assert torch.equal(
        p.evaluate_feature_conv1d(sample, x), p.evaluate_feature_conv1d(sample.function, x)
    )


def test_custody_tamper_and_complete_required_fields():
    sample = p.sample_feature_conv1d(
        request(), named_rng=np.random.default_rng(3), raw_rng=np.random.default_rng(7)
    )
    raw = json.loads(json.dumps(asdict(sample)))
    variants: list[dict[str, Any]] = []
    for key in raw:
        bad = copy.deepcopy(raw)
        del bad[key]
        variants.append(bad)
    bad = copy.deepcopy(raw)
    bad["raw"][3]["values"][0] += 0.1
    variants.append(bad)
    bad = copy.deepcopy(raw)
    bad["function"]["layers"][0]["bias"][0] += 0.1
    variants.append(bad)
    for bad in variants:
        with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
            p.feature_conv1d_from_dict(bad)
    with pytest.raises(ValueError, match="replay"):
        p.evaluate_feature_conv1d(
            replace(
                sample,
                function=replace(
                    sample.function, layers=(replace(sample.function.layers[0], bias=(0.0, 0.0)),)
                ),
            ),
            torch.ones((1, 10), dtype=torch.float64),
        )


def test_raw_preflight_atomic_stream_failure_and_caller_limits():
    good = request()
    with (
        patch("copy.deepcopy", side_effect=AssertionError("RNG clone before admission")),
        pytest.raises(ValueError, match="raw"),
    ):
        replace(good, raw_limits=replace(good.raw_limits, numbers=1))
    named, raw = np.random.default_rng(3), np.random.default_rng(7)
    before = (copy.deepcopy(named.bit_generator.state), copy.deepcopy(raw.bit_generator.state))
    extreme = replace(
        good, layers=(replace(good.layers[0], kernel_scale=1e308, kernel_mean=1e308, fan_in=False),)
    )
    with pytest.raises(ValueError, match="finite"):
        p.sample_feature_conv1d(extreme, named_rng=named, raw_rng=raw)
    assert (named.bit_generator.state, raw.bit_generator.state) == before
    sample = p.sample_feature_conv1d(good, named_rng=named, raw_rng=raw)
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.feature_conv1d_from_dict(asdict(sample), limits=p.CausalLimits(node_channels=2))
    with (
        patch("torch.nn.functional.conv1d", side_effect=AssertionError("backend called")),
        pytest.raises(ValueError, match="convolution|network|limit|metadata"),
    ):
        p.evaluate_feature_conv1d(
            sample.function,
            torch.ones((1000, 10), dtype=torch.float64),
            limits=p.CausalLimits(working_bytes=1024),
        )


def test_mixed_catalog_stack_choices_and_depthwise_extension():
    first = p.Conv1DLayerPrior(
        (
            p.Conv1DGeometry(2, 2, padding="same-upper", groups=2),
            p.Conv1DGeometry(4, 3, padding="valid", groups=2),
        ),
        activations=(p.CausalActivation(), p.CausalActivation("relu")),
        kernel_laws=("normal", "uniform"),
        bias_law="rademacher",
    )
    second = replace(
        first,
        geometries=(
            p.Conv1DGeometry(2, 1, groups=2),
            p.Conv1DGeometry(4, 2, padding="same-upper", groups=2),
        ),
    )
    prior = p.FeatureConv1DRequest(7, 2, (first, second), layout="position-major")
    named, raw = np.random.default_rng(9), np.random.default_rng(19)
    sample = p.sample_feature_conv1d(prior, named_rng=named, raw_rng=raw)
    oracle = np.random.default_rng(9)
    assert sample.choices == tuple(
        tuple(int(oracle.integers(2)) for _ in range(3)) for _ in range(2)
    )
    sample.validate_replay()
    assert len(sample.raw) == 10
    x = torch.arange(28, dtype=torch.float64).reshape(2, 14)
    actual = p.evaluate_feature_conv1d(sample, x)
    for i in range(2):
        assert torch.allclose(
            p.evaluate_feature_conv1d(sample, x[i : i + 1]),
            actual[i : i + 1],
            atol=1e-13,
            rtol=1e-13,
        )


@pytest.mark.parametrize("bad", [None, [], {}, {"version": "unknown"}])
def test_malformed_public_convolution_decoders(bad: Any):
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.feature_conv1d_from_dict(bad)
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.feature_conv1d_request_from_dict(bad)


def test_padding_geometry_and_alias_stream_rejection():
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.FeatureConv1DRequest(2, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 3),)),))
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.FeatureConv1DRequest(5, 3, (p.Conv1DLayerPrior((p.Conv1DGeometry(2, 1, groups=2),)),))
    with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
        p.Conv1DGeometry(1, 1, padding="same-upper", explicit_padding=(1, 1))
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="distinct"):
        p.sample_feature_conv1d(request(), named_rng=rng, raw_rng=rng)
    assert rng.bit_generator.state == before


def test_authored_metadata_padding_workspace_and_no_training_graph():
    layer = p.FeatureConv1DLayer(
        p.Conv1DGeometry(1, 3, dilation=2**30, padding="same-upper"), (((1.0, 2.0, 3.0),),), (0.0,)
    )
    function = p.FrozenFeatureConv1D(3, 1, (layer,))
    with (
        patch(
            "torch.nn.functional.pad",
            side_effect=AssertionError("padding allocated before admission"),
        ),
        pytest.raises(ValueError, match="memory"),
    ):
        p.evaluate_feature_conv1d(function, torch.ones((1, 3), dtype=torch.float64))
    simple = p.FrozenFeatureConv1D(
        3, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1), (((1.0,),),), (0.0,)),)
    )
    with pytest.raises(ValueError, match="metadata"):
        p.admit_feature_conv1d(simple, 1, limits=p.CausalLimits(parameters=3))
    values = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    output = p.evaluate_feature_conv1d(simple, values)
    assert not output.requires_grad
    assert torch.equal(output, values)


@pytest.mark.parametrize("cap", ["draws", "numbers", "working_bytes"])
def test_every_raw_cap_precedes_rng_copy(cap: str):
    prior = request()
    with (
        patch("copy.deepcopy", side_effect=AssertionError("RNG copy before raw admission")),
        pytest.raises(ValueError, match="raw"),
    ):
        replace(prior, raw_limits=replace(prior.raw_limits, **{cap: 1}))


def test_required_nested_archive_fields_are_never_defaulted():
    sample = p.sample_feature_conv1d(
        request(), named_rng=np.random.default_rng(3), raw_rng=np.random.default_rng(7)
    )
    for path in (
        ("request", "layers", 0),
        ("request", "layers", 0, "geometries", 0),
        ("function", "layers", 0),
        ("function", "layers", 0, "activation"),
        ("raw", 3),
        ("initial", 0),
        ("final", 1),
    ):
        original = json.loads(json.dumps(asdict(sample)))
        target = original
        for key in path:
            target = target[key]
        for missing in target:
            bad = copy.deepcopy(original)
            selected = bad
            for key in path:
                selected = selected[key]
            del selected[missing]
            with pytest.raises(ValueError, match="convolution|network|limit|metadata"):
                p.feature_conv1d_from_dict(bad)
