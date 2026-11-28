"""Frozen structural forest envelope; class outputs are majority-label indicators."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.recursive_forest_records import RecursiveForest
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest


@dataclass(frozen=True, slots=True)
class FrozenRecursiveForestFunction:
    source: RecursiveForest | SampledRecursiveForest
    version: str = "frozen-recursive-forest-function-v1"

    def __post_init__(self) -> None:
        if self.version != "frozen-recursive-forest-function-v1" or type(self.source) not in (
            RecursiveForest,
            SampledRecursiveForest,
        ):
            raise ValueError(
                "recursive function needs an authored forest or complete sampled receipt"
            )
        if isinstance(self.source, SampledRecursiveForest) and (
            self.source.plan.lifecycle != "frozen-function" or len(self.source.forests) != 1
        ):
            raise ValueError("per-row reference forests are not frozen structural functions")

    @property
    def forest(self) -> RecursiveForest:
        return (
            self.source.forests[0]
            if isinstance(self.source, SampledRecursiveForest)
            else self.source
        )

    @property
    def input_width(self) -> int:
        return self.forest.input_width

    @property
    def output_width(self) -> int:
        return (
            self.forest.classes
            if self.forest.task == "classification"
            else self.forest.output_width
        )

    @property
    def output_semantics(self) -> str:
        return (
            "majority-label-indicator"
            if self.forest.task == "classification"
            else "mean-regression-vector"
        )

    @property
    def parameter_count(self) -> int:
        # Full archive admission also counts Python containers and raw/snapshot metadata.
        count = sum(9 + len(n.value) for t in self.forest.trees for n in t.nodes)
        if isinstance(self.source, SampledRecursiveForest):
            count += sum(len(r.values) + 16 for r in self.source.raw)
        return count

    def admit(self, rows: int, limits: CausalLimits) -> dict[str, int]:
        report = self.forest.admit(rows, limits)
        numbers, storage = function_record_cost(self, limits)
        working = report["estimated_working_bytes"] + 4 * storage + 8 * rows * self.output_width
        if isinstance(self.source, SampledRecursiveForest):
            replay = self.source.plan.admit(limits, self.source.raw_limits)
            working += replay["estimated_working_bytes"]
            numbers += replay["parameters"]
        if (
            self.output_width > limits.node_channels
            or working > limits.working_bytes
            or numbers > limits.parameters
        ):
            raise ValueError("recursive structural function aggregate workspace exceeds limits")
        return {"parameters": numbers, "estimated_working_bytes": working}


def recursive_function_members(function: object) -> tuple[FrozenRecursiveForestFunction, ...]:
    """The only composite admitting this family is explicit nonrecursive parent composition."""
    from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction

    if type(function) is FrozenRecursiveForestFunction:
        return (function,)
    if type(function) is FrozenParentFunction:
        return tuple(
            c
            for c in (*function.children, *((function.joint,) if function.joint else ()))
            if isinstance(c, FrozenRecursiveForestFunction)
        )
    return ()
