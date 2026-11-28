"""Literal NumPy equations for the sampled neural-reducer verification fixtures."""

import numpy as np
from numpy.typing import NDArray

from r0.modules import priors as p


def activation(value: NDArray[np.float64], a: p.CalibratedActivation) -> NDArray[np.float64]:
    z = (
        (value - np.asarray(a.input_mean)) * np.asarray(a.input_inverse_std) + np.asarray(a.shift)
    ) * a.scale
    if a.kind == "identity":
        y = z
    elif a.kind == "silu":
        exp = np.exp(-np.abs(z))
        y = z * np.where(z >= 0, 1 / (1 + exp), exp / (1 + exp))
    elif a.kind == "softmax":
        exp = np.exp(z - z.max(axis=1, keepdims=True))
        y = exp / exp.sum(axis=1, keepdims=True)
    elif a.kind == "sign_positive":
        y = np.where(z >= 0, 1.0, -1.0)
    elif a.kind == "argsort":
        y = np.argsort(z, axis=1, kind="stable").astype(np.float64)
    elif a.kind == "signed_power":
        y = np.sign(z) * np.abs(z) ** dict(a.parameters)["exponent"]
    else:
        raise AssertionError(f"fixture oracle does not cover {a.kind}")
    return (y - np.asarray(a.output_mean)) * np.asarray(a.output_inverse_std)


def neural_parent(
    function: p.FrozenParentFunction, inputs: NDArray[np.float64]
) -> NDArray[np.float64]:
    members: list[NDArray[np.float64]] = []
    if function.joint is not None:
        assert isinstance(function.joint, p.FrozenLinearFunction)
        members.append(inputs @ np.asarray(function.joint.matrix).T)
    cursor = 0
    for member in function.children:
        assert isinstance(member, p.FrozenLinearFunction)
        members.append(
            inputs[:, cursor : cursor + member.input_width] @ np.asarray(member.matrix).T
        )
        cursor += member.input_width
    value = np.concatenate(members, axis=1)
    reducer = function.reducer
    assert reducer is not None
    stages = iter(reducer.activations)
    if reducer.start_activation:
        value = activation(value, next(stages))
    for index, linear in enumerate(reducer.linears):
        value = value @ np.asarray(linear.matrix).T
        if index + 1 < len(reducer.linears) or reducer.end_activation:
            value = activation(value, next(stages))
    assert next(stages, None) is None
    return value
