"""Explicit internal-noise equations, kept distinct from fixed structural masks."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse


@dataclass(frozen=True, slots=True)
class NetworkInnovationStage:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    placement: str = "post-block"
    exogenous: ExogenousDistribution = ExogenousDistribution()
    response: NoiseResponse = NoiseResponse()

    def __post_init__(self) -> None:
        if (
            type(self.means) is not tuple
            or type(self.scales) is not tuple
            or not self.means
            or len(self.means) != len(self.scales)
            or any(not finite_number(v) for v in (*self.means, *self.scales))
            or any(v < 0 for v in self.scales)
            or self.placement not in ("post-block", "pre-activation")
            or type(self.exogenous) is not ExogenousDistribution
            or type(self.response) is not NoiseResponse
        ):
            raise ValueError(
                "network innovation requires finite aligned means/scales and placement"
            )
        self.exogenous.validate_width(len(self.scales))


@dataclass(frozen=True, slots=True)
class NetworkInnovationPlan:
    stages: tuple[NetworkInnovationStage | None, ...]
    importance: tuple[float, ...] = ()
    importance_gates: bool = False
    profile: str = "r0-explicit"
    version: str = "network-innovation-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-innovation-plan-v1"
            or self.profile
            not in (
                "r0-explicit",
                "p1-paper-placement",
                "p1-778f0d0-placement",
                "tabicl-9530bd6-placement",
            )
            or type(self.stages) is not tuple
            or not self.stages
            or any(s is not None and type(s) is not NetworkInnovationStage for s in self.stages)
            or type(self.importance) is not tuple
            or any(not finite_number(v) for v in self.importance)
            or type(self.importance_gates) is not bool
            or self.importance_gates
            and not self.importance
        ):
            raise ValueError("invalid frozen network innovation plan")
        source = self.profile != "r0-explicit"
        if source and self.importance_gates:
            raise ValueError("gate importance is an r0 extension")
        for stage in self.stages:
            if stage is None:
                continue
            if source and (
                stage.exogenous != ExogenousDistribution() or stage.response != NoiseResponse()
            ):
                raise ValueError(
                    "source placement profiles require additive independent standard Gaussian draws"
                )
            if self.profile == "p1-paper-placement" and stage.placement != "pre-activation":
                raise ValueError("paper noise is inside activation")
            if self.profile in ("p1-778f0d0-placement", "tabicl-9530bd6-placement") and (
                stage.placement != "post-block" or any(stage.means)
            ):
                raise ValueError("pinned code uses centered post-affine Gaussian noise")
        if (
            self.profile in ("p1-778f0d0-placement", "tabicl-9530bd6-placement")
            and self.stages[0] is not None
        ):
            raise ValueError("pinned code exempts the first affine layer from noise")
