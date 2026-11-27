"""Caller-bounded reconstruction of complete operation ownership and its ledger."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_operation_registry import (
    JointOperationRegistry,
    JointOperationSpan,
    JointRegistryLimits,
)
from r0.modules.priors.lib.joint_sampling_records import joint_sampling_from_dict

_DEFAULT_LIMITS = CausalLimits()
_DEFAULT_REGISTRY_LIMITS = JointRegistryLimits()


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("joint registry records must be JSON objects")
    fields = cast(dict[str, Any], value)
    if set(fields) != {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("joint registry records must retain exactly their required fields")
    return dict(fields)


def joint_operation_registry_from_dict(
    data: dict[str, Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
    registry_limits: JointRegistryLimits = _DEFAULT_REGISTRY_LIMITS,
) -> JointOperationRegistry:
    """Payload limits cannot enlarge caller-owned byte/parameter/operation bounds."""
    if type(limits) is not CausalLimits or type(registry_limits) is not JointRegistryLimits:
        raise ValueError("joint registry reconstruction requires caller-owned frozen limits")
    admit_function_metadata(data, limits)
    fields = _record(data, JointOperationRegistry)
    if fields["version"] != "joint-operation-registry-v1":
        raise ValueError("unsupported joint registry version")
    raw = fields["operations"]
    if type(raw) not in (list, tuple):
        raise ValueError("joint registry operations must be an array")
    registry_limits.admit(len(raw), 0)
    owned_limits = JointRegistryLimits(**_record(fields["limits"], JointRegistryLimits))
    owned_limits.admit(len(raw), 0)
    operations: list[JointOperationSpan] = []
    path_bytes = 0
    for item in raw:
        span = _record(item, JointOperationSpan)
        path = span["operation"]
        if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
            raise ValueError("joint registry operation paths must be bounded arrays")
        span["operation"] = tuple(path)
        operation = JointOperationSpan(**span)
        path_bytes += operation.path_bytes
        registry_limits.admit(len(raw), path_bytes)
        owned_limits.admit(len(raw), path_bytes)
        operations.append(operation)
    try:
        decisions = joint_sampling_from_dict(fields["decisions"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "joint registry decision ledger has missing or malformed fields"
        ) from error
    return JointOperationRegistry(decisions, tuple(operations), owned_limits)
