"""Source-qualified law selection; parameter freezing is separate from normalization."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec

_FAMILIES = (
    "gaussian",
    "presampled_gaussian",
    "unit_uniform",
    "symmetric_uniform",
    "categorical",
    "zipf_clipped",
    "zipf_unbounded",
)


@dataclass(frozen=True, slots=True)
class SourceRootProfile:
    width: int
    selector: str = "p1-code-mixed"
    presample_gaussian: bool = False
    families: tuple[str, ...] = ()
    unbounded_exponent: float = 2.5
    p2_normal_probability: float = 0.5
    raw_limits: RawDrawLimits = RawDrawLimits()
    lifecycle: str = "frozen-selection-r0-v1"
    version: str = "source-root-profile-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "source-root-profile-v1"
            or self.lifecycle != "frozen-selection-r0-v1"
            or type(self.width) is not int
            or not 1 <= self.width <= 1024
            or self.selector
            not in (
                "p1-code-normal",
                "p1-code-uniform",
                "p1-code-mixed",
                "p2-node-mixture-r0",
                "forced-r0",
            )
            or type(self.presample_gaussian) is not bool
            or type(self.families) is not tuple
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid source root profile")
        if self.selector == "forced-r0":
            if len(self.families) != self.width or any(f not in _FAMILIES for f in self.families):
                raise ValueError("forced root requires one named family per coordinate")
        elif self.families:
            raise ValueError("unused forced root families")
        if self.presample_gaussian and not self.selector.startswith("p1-code-"):
            raise ValueError("presampling flag belongs to P1 code profiles")
        if not finite_number(self.unbounded_exponent) or self.unbounded_exponent <= 1:
            raise ValueError("unbounded Zipf exponent must exceed one")
        if "zipf_unbounded" not in self.families and self.unbounded_exponent != 2.5:
            raise ValueError("unused unbounded Zipf exponent")
        if (
            not finite_number(self.p2_normal_probability)
            or not 0 <= self.p2_normal_probability <= 1
        ):
            raise ValueError("P2 reconstruction probability must be in [0,1]")
        if self.selector != "p2-node-mixture-r0" and self.p2_normal_probability != 0.5:
            raise ValueError("unused P2 reconstruction probability")

    @property
    def gaussian_width(self) -> int:
        return self.width if self.presample_gaussian else self.families.count("presampled_gaussian")

    @property
    def source(self) -> str:
        if self.selector.startswith("p1-code-"):
            return "tabpfn-mlp-778f0d0-law-selection"
        if self.selector == "p2-node-mixture-r0":
            return "tabpfn2-node-normal-uniform-r0-probability"
        return "r0-forced-coordinate-laws"

    def worst_draws(self) -> tuple[RawDrawSpec, ...]:
        """Bound every conditional branch before sampling, including maximum K=10."""
        specs: list[RawDrawSpec] = []
        if self.gaussian_width:
            for role in ("gaussian_means", "gaussian_scale_factors"):
                specs.append(
                    RawDrawSpec(("root-profile", "model"), role, "normal", (self.gaussian_width,))
                )
        if self.selector == "p1-code-mixed":
            for role in ("zipf_threshold_unused", "categorical_threshold", "normal_threshold"):
                specs.append(RawDrawSpec(("root-profile", "selection"), role, "random", ()))
        if self.selector == "p2-node-mixture-r0":
            specs.append(RawDrawSpec(("root-profile", "selection"), "node_family", "random", ()))
        for i in range(self.width):
            owner = ("root-profile", f"coordinate:{i}")
            family = self.families[i] if self.families else ""
            mixed = self.selector == "p1-code-mixed"
            if mixed:
                for role in ("normal_test", "categorical_test"):
                    specs.append(RawDrawSpec(owner, role, "random", ()))
            if mixed or family == "categorical":
                specs.extend(
                    (
                        RawDrawSpec(owner, "category_count", "integers", (), (("high", 9),)),
                        RawDrawSpec(owner, "category_weights", "random", (10,)),
                    )
                )
            if mixed or family == "zipf_clipped":
                specs.append(RawDrawSpec(owner, "zipf_exponent", "random", ()))
        return tuple(specs)

    def admit(self, limits: CausalLimits) -> None:
        if type(limits) is not CausalLimits:
            raise ValueError("root profile requires causal limits")
        specs = self.worst_draws()
        numbers = sum(s.numbers for s in specs)
        estimate = limits.admit_dimensions(
            (self.width,),
            ((),),
            rows=1,
            calibration_rows=0,
            features=self.width,
            outputs=self.width,
            extra_parameters=24 * self.width + numbers + 16 * len(specs),
            extra_working_channels=4 * self.width,
        )
        # Four retained JSON snapshots plus simultaneous replay/transaction states.
        if (
            estimate["estimated_working_bytes"] + 8 * self.raw_limits.state_bytes
            > limits.working_bytes
        ):
            raise ValueError("root profile snapshot working limit exceeded")
