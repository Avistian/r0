"""JSON-number adaptation for explicit authored edits, never stored model replay."""

import math
from typing import Any, cast

from r0.modules import priors


def _numbers(value: Any, maximum: int) -> list[float]:
    if type(value) is not list or not 1 <= len(cast(list[Any], value)) <= maximum:
        raise ValueError("authored XGBoost numeric array exceeds geometry")
    result: list[float] = []
    for number in cast(list[Any], value):
        if type(number) not in (int, float):
            raise ValueError("authored XGBoost numbers must be finite int or float, not bool")
        try:
            converted = float(number)
        except OverflowError as exc:
            raise ValueError("authored XGBoost number is not finite") from exc
        if not math.isfinite(converted):
            raise ValueError("authored XGBoost number is not finite")
        result.append(converted)
    return result


def prepare_authored_xgboost(request: dict[str, Any], policy: object) -> None:
    """Adapt only declared real-valued fields inside authored node equations.

    JavaScript has one Number type. Its JSON serialization emits integer tokens for
    integral floats. This editor boundary assigns the explicit function schema's
    float types, including deliberately changed integer-valued leaves/thresholds.
    Node/child indices, bool routing flags and all archive decoders stay unchanged.
    The worker owns the decoded request; no caller-owned objects are mutated.
    """
    if policy != "authored-mechanism-v1":
        raise ValueError("mechanism edit requires the explicit authored recipe policy")
    recipe = request.get("recipe")
    if type(recipe) is not dict:
        return  # The ordinary request decoder reports malformed recipes.
    recipe = cast(dict[str, Any], recipe)
    if recipe.get("version") not in (
        "functional-causal-v1",
        "native-causal-v1",
        "native-vector-causal-v1",
        "converted-causal-v1",
    ):
        return
    nodes = recipe.get("nodes")
    if type(nodes) is not list:
        return
    limits = priors.CausalLimits(**request.get("causal_limits", {}))
    if len(cast(list[Any], nodes)) > limits.nodes:
        raise ValueError("authored equation count exceeds node limit")
    functions: list[dict[str, Any]] = []
    stack: list[Any] = [nodes]
    visited = 0
    while stack:
        value = stack.pop()
        visited += 1
        if visited > limits.parameters:
            raise ValueError("authored equation metadata exceeds parameter limit")
        if type(value) is list:
            items = cast(list[Any], value)
            if visited + len(stack) + len(items) > limits.parameters:
                raise ValueError("authored equation metadata exceeds parameter limit")
            stack.extend(items)
        elif type(value) is dict:
            fields = cast(dict[str, Any], value)
            if fields.get("version") == "frozen-xgboost-vector-v1":
                functions.append(fields)
            if visited + len(stack) + len(fields) > limits.parameters:
                raise ValueError("authored equation metadata exceeds parameter limit")
            stack.extend(fields.values())
    if functions and 64 * visited > limits.working_bytes:
        raise ValueError("authored XGBoost numeric adaptation exceeds working limit")
    # Complete metadata admission precedes numeric conversion or replacement arrays.
    for fields in functions:
        fields["initial"] = _numbers(fields.get("initial"), 100)
        trees = fields.get("trees")
        if type(trees) is not list or not 1 <= len(cast(list[Any], trees)) <= 32:
            raise ValueError("authored XGBoost requires bounded tree arrays")
        for item in cast(list[Any], trees):
            if type(item) is not dict:
                raise ValueError("authored XGBoost tree must be an object")
            tree = cast(dict[str, Any], item)
            tree["thresholds"] = _numbers(tree.get("thresholds"), 511)
            rows = tree.get("values")
            if type(rows) is not list or not 1 <= len(cast(list[Any], rows)) <= 511:
                raise ValueError("authored XGBoost leaf rows exceed geometry")
            tree["values"] = [_numbers(row, 100) for row in cast(list[Any], rows)]
