"""Declared stage placement, preference policy and independent calibration protocol."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class SourceActivationPrior:
    selector: str = "standard"
    mode: str = "meta"
    policy: str = "shared"
    placement: str = "output"
    noise_after_probability: float = 0.5
    calibration_rows: int = 256
    version: str = "source-activation-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "source-activation-prior-v1" or self.selector not in (
            "standard",
            "plain",
        ):
            raise ValueError("unsupported source activation prior or selector")
        if self.mode not in ("local", "global", "meta") or self.policy not in ("shared", "refresh"):
            raise ValueError("unsupported source activation preference policy")
        if self.placement not in ("input", "output", "both"):
            raise ValueError("source activation placement must be input, output or both")
        if (
            not finite_number(self.noise_after_probability)
            or not 0 <= self.noise_after_probability <= 1
        ):
            raise ValueError("source noise-after probability must be in [0,1]")
        if type(self.calibration_rows) is not int or not 1 <= self.calibration_rows < 2**63:
            raise ValueError("source calibration rows must be positive and below 2**63")


def rescale_request() -> NumericalParameter:
    return NumericalParameter("rr_scale", 1.0, 10.0, scale="log")


@dataclass(frozen=True, slots=True)
class SourceStageBinding:
    node: int
    stage: str
    selection: int
    rescale_draw: int | None = None

    def __post_init__(self) -> None:
        if any(type(i) is not int or i < 0 for i in (self.node, self.selection)):
            raise ValueError("source binding indices must be nonnegative integers")
        if self.stage not in ("input", "output"):
            raise ValueError("source binding stage must be input or output")
        if self.rescale_draw is not None and (
            type(self.rescale_draw) is not int or self.rescale_draw < 0
        ):
            raise ValueError("source binding rescale reference must be nonnegative")

    @property
    def parameter_count(self) -> int:
        return 2 + int(self.rescale_draw is not None)


def source_stages(
    parents: tuple[tuple[int, ...], ...], prior: SourceActivationPrior
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (i, stage)
        for i, incoming in enumerate(parents)
        for stage in ("input", "output")
        if (
            bool(incoming) and prior.placement in ("input", "both")
            if stage == "input"
            else prior.placement in ("output", "both")
        )
    )


SOURCE_ACTIVATION_PROFILES = {
    f"causal-source-{selector}{suffix}-v1": SourceActivationPrior(
        selector=selector, mode=mode, policy=policy
    )
    for selector in ("standard", "plain")
    for suffix, mode, policy in (
        ("", "meta", "shared"),
        ("-refreshed", "meta", "refresh"),
        ("-local", "local", "shared"),
        ("-global", "global", "shared"),
    )
}
