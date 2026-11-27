"""Bounded JSON reconstruction for all eight frozen function families."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.calibrated_activations import CalibratedActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
from r0.modules.priors.lib.fitted_tree_decoding import decode_fitted_tree
from r0.modules.priors.lib.fitted_tree_records import FrozenFittedTreeFunction
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
    FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction, admit_frozen_function
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)
from r0.modules.priors.lib.frozen_kumo_product import FrozenKumoProductFunction
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.network_block_function import FrozenNetworkBlockFunction
from r0.modules.priors.lib.network_decoding import decode_block, decode_network
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction

_DEFAULT_LIMITS = CausalLimits()
_REGISTRY = {
    "frozen-feature-conv1d-r0-v1": FrozenFeatureConv1D,
    "frozen-network-block-v1": FrozenNetworkBlockFunction,
    "frozen-structural-network-v1": FrozenNetworkFunction,
    "frozen-fitted-tree-function-v1": FrozenFittedTreeFunction,
    "frozen-linear-function-v1": FrozenLinearFunction,
    "frozen-quadratic-function-v1": FrozenQuadraticFunction,
    "frozen-oblivious-tree-function-v1": FrozenTreeFunction,
    "frozen-rff-function-v1": FrozenRFFFunction,
    "frozen-discretization-function-v1": FrozenDiscretizationFunction,
    "frozen-em-function-v1": FrozenEMFunction,
    "frozen-product-function-v1": FrozenProductFunction,
    "frozen-kumo-product-function-v1": FrozenKumoProductFunction,
    "frozen-mlp-function-v1": FrozenMLPFunction,
    "frozen-parent-function-v1": FrozenParentFunction,
}
_LINEAR = frozenset(("frozen-linear-function-v1",))
_KUMO = frozenset(
    (
        "frozen-linear-function-v1",
        "frozen-quadratic-function-v1",
        "frozen-oblivious-tree-function-v1",
        "frozen-rff-function-v1",
        "frozen-discretization-function-v1",
        "frozen-em-function-v1",
        "frozen-mlp-function-v1",
    )
)
_CHEAP = frozenset(_REGISTRY) - {
    "frozen-feature-conv1d-r0-v1",
    "frozen-network-block-v1",
    "frozen-structural-network-v1",
    "frozen-kumo-product-function-v1",
    "frozen-parent-function-v1",
    "frozen-fitted-tree-function-v1",
    "frozen-em-function-v1",
    "frozen-product-function-v1",
    "frozen-mlp-function-v1",
}


def _record(data: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(data) is not dict:
        raise ValueError("frozen function record must be an object")
    mapping = cast(dict[str, Any], data)
    if set(mapping) - {f.name for f in dataclass_fields(record_type)}:
        raise ValueError("frozen function record contains unknown fields")
    return dict(mapping)


def _array(value: Any) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("frozen function arrays must be lists or tuples")
    return cast(list[Any] | tuple[Any, ...], value)


def _matrix(value: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(_array(row)) for row in _array(value))


def _activation(data: Any) -> CalibratedActivation:
    fields = _record(data, CalibratedActivation)
    if fields.get("calibration_version") != "independent-standard-activation-v1":
        raise ValueError("MLP requires frozen standard activation records")
    parameters = _array(fields.get("parameters", ()))
    # Authored calibrated activations also accept four affine modifiers and up
    # to two primitive parameters; decoding must preserve every accepted record.
    if len(parameters) > 6 or any(len(_array(pair)) != 2 for pair in parameters):
        raise ValueError("MLP activation parameters require at most six two-element arrays")
    fields["parameters"] = tuple(tuple(pair) for pair in parameters)
    for name in (
        "input_mean",
        "input_inverse_std",
        "shift",
        "output_mean",
        "output_inverse_std",
        "shift_rows",
    ):
        fields[name] = tuple(_array(fields[name]))
    return CalibratedActivation(**fields)


def _decode(data: Any, allowed: frozenset[str]) -> FrozenFunction:
    if type(data) is not dict:
        raise ValueError("frozen function record must be an object")
    raw = cast(dict[str, Any], data)
    version = raw.get("version")
    if type(version) is not str or version not in allowed:
        raise ValueError("unsupported or recursively disallowed frozen function version")
    record_type = _REGISTRY[version]
    fields = _record(raw, record_type)
    if record_type is FrozenFeatureConv1D:
        from r0.modules.priors.lib.feature_conv_decoding import decode_feature_conv1d

        return decode_feature_conv1d(fields)
    if record_type is FrozenNetworkBlockFunction:
        fields["incoming_widths"] = tuple(_array(fields["incoming_widths"]))
        fields["block"] = decode_block(fields["block"])
        return FrozenNetworkBlockFunction(**fields)
    if record_type is FrozenNetworkFunction:
        return decode_network(fields)
    if record_type is FrozenFittedTreeFunction:
        return decode_fitted_tree(fields)
    if record_type is FrozenParentFunction:
        members = frozenset(_REGISTRY) - {"frozen-parent-function-v1"}
        fields["children"] = tuple(_decode(c, members) for c in _array(fields["children"]))
        fields["joint"] = (
            _decode(fields["joint"], members) if fields.get("joint") is not None else None
        )
        fields["reducer"] = (
            _decode(fields["reducer"], frozenset(("frozen-mlp-function-v1",)))
            if fields.get("reducer") is not None
            else None
        )
        fields["weights"] = tuple(_array(fields.get("weights", ())))
    elif record_type is FrozenLinearFunction:
        fields["matrix"] = _matrix(fields["matrix"])
    elif record_type is FrozenQuadraticFunction:
        fields["indices"] = tuple(_array(fields["indices"]))
        fields["matrices"] = tuple(_matrix(m) for m in _array(fields["matrices"]))
    elif record_type is FrozenTreeFunction:
        fields["features"] = _matrix(fields["features"])
        fields["thresholds"] = _matrix(fields["thresholds"])
        fields["leaves"] = tuple(_matrix(table) for table in _array(fields["leaves"]))
    elif record_type is FrozenRFFFunction:
        fields["frequencies"] = _matrix(fields["frequencies"])
        fields["coefficients"] = _matrix(fields["coefficients"])
        fields["phases"] = tuple(_array(fields["phases"]))
    elif record_type in (FrozenDiscretizationFunction, FrozenEMFunction):
        fields["centers"] = _matrix(fields["centers"])
        fields["linear"] = _decode(fields["linear"], _LINEAR)
        if record_type is FrozenEMFunction:
            fields["scales"] = tuple(_array(fields["scales"]))
    elif record_type in (FrozenProductFunction, FrozenKumoProductFunction):
        allowed_children = _KUMO if record_type is FrozenKumoProductFunction else _CHEAP
        fields["left"] = _decode(fields["left"], allowed_children)
        fields["right"] = _decode(fields["right"], allowed_children)
    else:
        fields["linears"] = tuple(_decode(layer, _LINEAR) for layer in _array(fields["linears"]))
        fields["activations"] = tuple(_activation(a) for a in _array(fields.get("activations", ())))
    return record_type(**fields)


def decode_admitted_function(data: Any) -> FrozenFunction:
    """Decode after enclosing raw metadata admission and before aggregate geometry admission.

    Graph nodes are not the virtual dimensions counted by standalone admission.
    Enclosing callers must admit the complete graph before allocating numerical arrays.
    """
    return _decode(data, frozenset(_REGISTRY))


def frozen_function_from_dict(
    data: dict[str, Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> FrozenFunction:
    if type(limits) is not CausalLimits:
        raise ValueError("frozen function decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    function = decode_admitted_function(data)
    # Admission for one evaluation row includes center-dependent working floors.
    # Every later call separately admits its actual row count before tensors.
    admit_frozen_function(function, 1, limits)
    return function
