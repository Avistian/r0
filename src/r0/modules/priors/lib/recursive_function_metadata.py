"""Separate structural construction custody and conditional-on-forest moment meaning."""

from typing import Any

from r0.modules.priors.lib.recursive_forest_function import recursive_function_members
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest


def recursive_mechanism_metadata(functions: tuple[object, ...]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for node, function in enumerate(functions):
        for member, frozen in enumerate(recursive_function_members(function)):
            source = frozen.source
            entries.append(
                {
                    "node": node,
                    "member": member,
                    "output_semantics": frozen.output_semantics,
                    "lifecycle": "frozen-function",
                    "construction": "sampled-with-complete-raw-receipt"
                    if isinstance(source, SampledRecursiveForest)
                    else "authored",
                    "source_profile": source.plan.source
                    if isinstance(source, SampledRecursiveForest)
                    else "authored-no-sampling-law-claim",
                    "moment_scope": (
                        "bounded conditional on retained finite leaves; "
                        "not an unconditional construction-population claim"
                    ),
                    "traversal": (
                        "float64 inclusive-left; CPU literal paths; "
                        "result returned on requested tensor device"
                    ),
                }
            )
    return {"recursive_forests": entries} if entries else {}
