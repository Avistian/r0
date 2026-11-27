"""Strict registry-owned JSON shapes before legacy preference reconstruction."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.categorical_preferences import (
    CategoricalDraw,
    CategoricalParameter,
    CategoricalPreference,
    CategoricalPreferences,
)
from r0.modules.priors.lib.joint_sampling_records import JointDecision, JointSamplingRecord
from r0.modules.priors.lib.parameter_preferences import (
    NumericalDraw,
    NumericalParameter,
    NumericalPreference,
    ParameterPreferences,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("joint ledger records must be JSON objects")
    record = cast(dict[str, Any], value)
    if set(record) != {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("joint ledger records must retain exactly their required fields")
    return record


def _array(value: Any) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("joint ledger sequence fields must be JSON arrays")
    return cast(list[Any] | tuple[Any, ...], value)


def _numerical(value: Any) -> None:
    record = _record(value, ParameterPreferences)
    _record(record["limits"], PreferenceLimits)
    for key, kind in (("preferences", NumericalPreference), ("draws", NumericalDraw)):
        for raw in _array(record[key]):
            item = _record(raw, kind)
            request = _record(item["request"], NumericalParameter)
            _array(request["shape"])
            _array(item["units"])
            if key == "draws":
                _array(item["values"])


def validate_joint_ledger_shape(value: Any) -> None:
    """The enclosing decoder admits bytes/leaves/depth before this structural pass.

    Keep legacy standalone decoder behavior separate. A complete registry always
    persists all ledger fields, including empty collections and default values.
    Numeric law and reference validation still belongs to the typed records.
    """
    ledger = _record(value, JointSamplingRecord)
    _record(ledger["limits"], PreferenceLimits)
    for raw in _array(ledger["events"]):
        event = _record(raw, JointDecision)
        _array(event["operation"])
    _numerical(ledger["numerical"])
    categorical = _record(ledger["categorical"], CategoricalPreferences)
    _record(categorical["limits"], PreferenceLimits)
    for key, kind in (("preferences", CategoricalPreference), ("draws", CategoricalDraw)):
        for raw in _array(categorical[key]):
            item = _record(raw, kind)
            request = _record(item["request"], CategoricalParameter)
            _array(request["choices"])
            if key == "preferences":
                if item["hyperparameters"] is not None:
                    _numerical(item["hyperparameters"])
                if item["weights"] is not None:
                    weights = _record(item["weights"], PolynomialWeightVector)
                    _array(weights["draws"])
                    _array(weights["permutation"])
