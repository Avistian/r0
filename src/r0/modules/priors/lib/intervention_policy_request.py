"""Source selection constraints and explicit reconstruction controls for interventions."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.intervention_policy_laws import (
    InterventionParameterLaw,
    numeric_leaves,
    parameter_group,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class InterventionPolicyRequest:
    base: FunctionalCausalRecipe
    label_nodes: tuple[int, ...]
    kind: str = "hard"
    profile: str = "taffy-selection-r0-laws-v1"
    eligibility: tuple[int, ...] = ()
    count: int | None = None
    fraction: float = 0.05
    location: str = "state_and_readout"
    mode: str = "population"
    population: int = 1
    bounds_lifecycle: str = "independent-calibration-r0"
    reference_rows: int = 64
    reference_seed: int = 0
    reference_task_id: int = 0
    parameter_laws: tuple[InterventionParameterLaw, ...] = ()
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "intervention-policy-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "intervention-policy-request-v1"
            or type(self.base) is not FunctionalCausalRecipe
            or self.kind not in ("hard", "soft")
            or self.profile not in ("taffy-selection-r0-laws-v1", "explicit-policy-r0-v1")
            or self.location not in ("state", "readout", "state_and_readout")
            or self.mode not in ("matched", "population")
            or self.bounds_lifecycle
            not in ("independent-calibration-r0", "source-reference-population")
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.fraction) not in (int, float)
            or not math.isfinite(self.fraction)
            or not 0 < self.fraction <= 1
            or self.count is not None
            and (type(self.count) is not int or self.count < 1)
            or any(
                type(v) is not int or not 0 <= v < 2**63
                for v in (self.reference_seed, self.reference_task_id)
            )
            or type(self.population) is not int
            or not 1 <= self.population < 2**63
            or type(self.reference_rows) is not int
            or not 1 <= self.reference_rows <= 20000
        ):
            raise ValueError("invalid sampled intervention policy")
        for values in (self.label_nodes, self.eligibility):
            if (
                type(values) is not tuple
                or len(set(values)) != len(values)
                or any(type(i) is not int or not 0 <= i < len(self.base.nodes) for i in values)
            ):
                raise ValueError("policy nodes require unique valid exact indices")
        if not self.label_nodes or set(self.label_nodes) != {i for i, _ in self.base.target}:
            raise ValueError("label exclusions must equal the retained target-node set")
        if (
            set(self.eligible) & set(self.label_nodes)
            or not self.eligible
            or self.selected_count > len(self.eligible)
        ):
            raise ValueError(
                "policy eligibility/count cannot intervene on labels or be silently repaired"
            )
        if self.profile == "taffy-selection-r0-laws-v1" and (
            len(self.label_nodes) != 1
            or self.eligibility
            or self.count is not None
            or self.fraction != 0.05
            or self.mode != "population"
            or self.location != "state_and_readout"
            or any(n.function.output_width != 1 or n.pipeline is not None for n in self.base.nodes)
        ):
            raise ValueError(
                "TAFFY selection profile requires scalar nodes, one label and fresh populations"
            )
        if (
            self.bounds_lifecycle == "source-reference-population"
            and self.reference_rows != self.base.task.rows
        ):
            raise ValueError("source reference bounds must retain the complete original population")
        if type(self.parameter_laws) is not tuple or any(
            type(x) is not InterventionParameterLaw for x in self.parameter_laws
        ):
            raise ValueError("soft laws must be frozen explicit records")
        if self.kind == "soft" and self.location != "state_and_readout":
            raise ValueError(
                "soft function resampling acts before pipeline/readout; no readout-only alias"
            )
        if self.kind == "hard" and self.parameter_laws:
            raise ValueError("hard policies do not consume soft parameter laws")
        paths: list[tuple[int, tuple[str, ...]]] = []
        for law in self.parameter_laws:
            if law.node not in self.eligible:
                raise ValueError("parameter law must belong to an eligible node")
            for node, path in paths:
                if node == law.node and (
                    path[: len(law.path)] == law.path or law.path[: len(path)] == path
                ):
                    raise ValueError("parameter paths must be nonoverlapping")
            paths.append((law.node, law.path))
            try:
                numeric_leaves(parameter_group(self.base.nodes[law.node].function, law.path)[2])
            except (KeyError, IndexError, AttributeError, TypeError, ValueError) as error:
                raise ValueError("invalid resampled function parameter path") from error
        if self.kind == "soft" and {x.node for x in self.parameter_laws} != set(self.eligible):
            raise ValueError("every eligible soft mechanism needs an explicit parameter law")

    @property
    def eligible(self) -> tuple[int, ...]:
        return self.eligibility or tuple(
            i for i in range(len(self.base.nodes)) if i not in self.label_nodes
        )

    @property
    def selected_count(self) -> int:
        return (
            self.count if self.count is not None else math.ceil(self.fraction * len(self.eligible))
        )
