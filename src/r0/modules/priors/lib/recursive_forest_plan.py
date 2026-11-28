"""Explicit MITRA recursive-tree controls and conservative construction admission."""

import math
from dataclasses import dataclass, fields

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class RecursiveForestPlan:
    input_width: int = 2
    estimators: int = 3
    min_depth: int = 1
    max_depth: int = 4
    terminal_probability: float = 0.2
    intervals: tuple[tuple[float, float], ...] = ((-3.0, 3.0), (-3.0, 3.0))
    task: str = "regression"
    leaf_mean: tuple[float, ...] = (0.0,)
    leaf_scale: tuple[float, ...] = (1.0,)
    classes: int = 0
    class_assignment: str = "none"
    tie_break: str = "none"
    split_probabilities: tuple[float, ...] = ()
    source: str = "mitra-prose-v1"
    lifecycle: str = "frozen-function"
    populations: int = 1
    version: str = "recursive-forest-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-forest-plan-v1"
            or type(self.input_width) is not int
            or not 1 <= self.input_width <= 256
            or type(self.estimators) is not int
            or not 1 <= self.estimators <= 64
            or type(self.min_depth) is not int
            or type(self.max_depth) is not int
            or not 0 <= self.min_depth <= self.max_depth <= 10
            or not finite_number(self.terminal_probability)
            or not 0 <= self.terminal_probability <= 1
            or self.source not in ("mitra-prose-v1", "mitra-algorithm4-v1", "r0-extension-v1")
            or self.lifecycle not in ("frozen-function", "fresh-forest-per-row")
            or type(self.populations) is not int
            or not 1 <= self.populations <= 4096
            or self.lifecycle == "frozen-function"
            and self.populations != 1
        ):
            raise ValueError("invalid recursive forest controls or lifecycle")
        if (
            type(self.intervals) is not tuple
            or len(self.intervals) != self.input_width
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(not finite_number(v) for v in pair)
                or pair[0] >= pair[1]
                or not math.isfinite(pair[1] - pair[0])
                for pair in self.intervals
            )
        ):
            raise ValueError("recursive forest requires finite ordered feature intervals")
        if (
            type(self.split_probabilities) is not tuple
            or self.split_probabilities
            and (
                len(self.split_probabilities) != self.input_width
                or any(not finite_number(v) or v < 0 for v in self.split_probabilities)
                or not math.isclose(math.fsum(self.split_probabilities), 1.0, abs_tol=1e-12)
            )
        ):
            raise ValueError("invalid recursive split probabilities")
        if self.task == "regression":
            if (
                type(self.leaf_mean) is not tuple
                or type(self.leaf_scale) is not tuple
                or not 1 <= len(self.leaf_mean) <= 32
                or len(self.leaf_mean) != len(self.leaf_scale)
                or any(not finite_number(v) for v in self.leaf_mean)
                or any(not finite_number(v) or v < 0 for v in self.leaf_scale)
                or type(self.classes) is not int
                or self.classes != 0
                or self.class_assignment != "none"
                or self.tie_break != "none"
            ):
                raise ValueError("regression requires Gaussian leaves and no class controls")
        elif self.task == "classification":
            if (
                type(self.classes) is not int
                or not 2 <= self.classes <= 100
                or self.leaf_mean != ()
                or self.leaf_scale != ()
                or type(self.leaf_mean) is not tuple
                or type(self.leaf_scale) is not tuple
                or self.class_assignment not in ("random-start-cycle", "zero-start-cycle", "iid")
                or self.tie_break not in ("smallest", "largest", "first-tree")
            ):
                raise ValueError("classification requires class labels and an explicit tie rule")
        else:
            raise ValueError("recursive task must be regression or classification")
        if self.source != "r0-extension-v1" and (
            self.min_depth < 1
            or self.split_probabilities
            or len(set(self.intervals)) != 1
            or self.output_width != 1
            or self.task == "classification"
            and self.class_assignment
            != ("random-start-cycle" if self.source == "mitra-prose-v1" else "zero-start-cycle")
        ):
            raise ValueError("source profile requires disclosed scalar/uniform/cyclic controls")

    @property
    def output_width(self) -> int:
        return len(self.leaf_mean) if self.task == "regression" else 1

    @property
    def worst_nodes(self) -> int:
        return self.populations * self.estimators * (2 ** (self.max_depth + 1) - 1)

    def admit(self, limits: CausalLimits, raw_limits: RawDrawLimits) -> dict[str, int]:
        """Bound worst geometry, live path copies, raw records and replay before RNG work."""
        if type(limits) is not CausalLimits or type(raw_limits) is not RawDrawLimits:
            raise ValueError("recursive forest admission requires explicit frozen limits")
        n, w = self.worst_nodes, self.output_width
        trees = self.populations * self.estimators
        draws = 4 * n + trees
        numbers = n * (3 + w) + trees
        # Every node has at most split, threshold, termination and vector leaf draws.
        # Includes complete raw descriptor keys, digests, path strings and derived records.
        raw_bytes = n * (12000 + 128 * w + 128 * self.input_width) + 3000 * trees
        parameters = n * (160 + 4 * w + 2 * self.input_width) + 256 * trees + 8 * self.input_width
        working = 4 * raw_bytes + 8 * raw_limits.state_bytes + 512 * parameters
        if (
            n > limits.nodes
            or max(self.input_width, w) > limits.node_channels
            or self.input_width + w > limits.total_channels
            or parameters > limits.parameters
            or working > limits.working_bytes
        ):
            raise ValueError("recursive forest worst-case geometry or workspace limit exceeded")
        if (
            draws > raw_limits.draws
            or numbers > raw_limits.numbers
            or raw_bytes + 2 * raw_limits.state_bytes > raw_limits.working_bytes
        ):
            raise ValueError("recursive forest worst-case raw limit exceeded")
        return {
            "nodes": n,
            "draws": draws,
            "numbers": numbers,
            "parameters": parameters,
            "estimated_working_bytes": working,
        }


def recursive_effective_limits(declared: CausalLimits, caller: CausalLimits) -> CausalLimits:
    if type(declared) is not CausalLimits or type(caller) is not CausalLimits:
        raise ValueError("recursive limits must be frozen causal limits")
    return CausalLimits(
        **{
            f.name: min(getattr(declared, f.name), getattr(caller, f.name))
            for f in fields(CausalLimits)
        }
    )
