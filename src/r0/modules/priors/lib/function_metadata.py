"""Bound raw function records before constructing typed plans or arrays."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits


def admit_function_metadata(data: Any, limits: CausalLimits) -> None:
    numbers, containers, strings, nodes = 0, 0, 0, 0
    recursive_envelopes: list[dict[str, Any]] = []
    sampled_convolutions: list[dict[str, Any]] = []
    task_graphs: list[dict[str, Any]] = []
    node_limit = 16 * limits.parameters + 64 * limits.nodes

    def visit(value: Any, depth: int) -> None:
        nonlocal numbers, containers, strings, nodes
        nodes += 1
        if depth > 24 or nodes > node_limit:
            raise ValueError("function raw traversal limit exceeded")
        if type(value) in (int, float, bool):
            numbers += 1
        elif type(value) is str:
            if len(value) > limits.working_bytes:
                raise ValueError("function raw text limit exceeded")
            strings += len(value.encode("utf-8"))
        elif type(value) in (dict, list, tuple):
            containers += 1
        elif value is not None:
            raise ValueError("function payload must contain JSON primitives")
        if (
            numbers > limits.parameters
            or 64 * numbers + 32 * containers + strings > limits.working_bytes
        ):
            raise ValueError("function raw parameter or estimated storage limit exceeded")
        if type(value) is dict:
            mapping = cast(dict[Any, Any], value)
            if "nodes" in mapping and "task" in mapping:
                task_graphs.append(mapping)
            if mapping.get("version") == "sampled-feature-conv1d-r0-v1":
                sampled_convolutions.append(mapping)
            if mapping.get("version") == "frozen-recursive-forest-function-v1":
                recursive_envelopes.append(mapping)
            if 2 * len(mapping) > node_limit - nodes:
                raise ValueError("function raw object limit exceeded")
            for key, item in mapping.items():
                if type(key) is not str:
                    raise ValueError("function object keys must be strings")
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif type(value) in (list, tuple):
            sequence = cast(list[Any] | tuple[Any, ...], value)
            if len(sequence) > node_limit - nodes:
                raise ValueError("function raw array limit exceeded")
            for item in sequence:
                visit(item, depth + 1)

    visit(data, 0)
    if recursive_envelopes or sampled_convolutions:
        from r0.modules.priors.lib.recursive_function_admission import admit_recursive_payloads

        admit_recursive_payloads(
            recursive_envelopes,
            storage=64 * numbers + 32 * containers + strings,
            numbers=numbers,
            limits=limits,
            graphs=task_graphs,
            convolutions=sampled_convolutions,
        )
