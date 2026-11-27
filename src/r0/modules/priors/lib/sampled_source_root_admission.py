"""Aggregate root selection and independent-fit admission before RNG state copies."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.source_root_calibration import validate_root_normalization_modes
from r0.modules.priors.lib.source_root_laws import SourceRootLaw
from r0.modules.priors.lib.source_root_profiles import SourceRootProfile
from r0.modules.priors.lib.source_root_sampling import source_root_draw_specs


def admit_sampled_source_root(
    profile: SourceRootProfile,
    law: SourceRootLaw,
    modes: tuple[str, ...],
    rows: int,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
) -> dict[str, int]:
    if (
        type(profile) is not SourceRootProfile
        or type(law) is not SourceRootLaw
        or law.width != profile.width
        or type(limits) is not CausalLimits
        or type(raw_limits) is not RawDrawLimits
        or type(rows) is not int
        or rows < 0
    ):
        raise ValueError("invalid sampled root construction controls")
    validate_root_normalization_modes(law, modes, max(rows, 1))
    fitted = any(mode != "identity" for mode in modes)
    if fitted != (rows > 0):
        raise ValueError("independent fit rows required exactly for nonidentity normalization")
    profile.admit(limits)
    selection = profile.worst_draws()
    draws = source_root_draw_specs(law, rows, ("source-root-calibration",)) if fitted else ()
    # Fixed raw stream name is retained by the outer calibration transaction.
    raw_bytes = sum(spec.retained_bytes("raw") for spec in draws)
    if fitted and (
        len(draws) > raw_limits.draws
        or sum(spec.numbers for spec in draws) > raw_limits.numbers
        or raw_bytes + 2 * raw_limits.state_bytes > raw_limits.working_bytes
    ):
        raise ValueError("sampled root aggregate calibration raw limit exceeded")
    # Simultaneously retained profile, its replay and calibration raw/frozen copies.
    parameters = (
        64 * law.width
        + 2 * sum(spec.numbers for spec in selection)
        + 64 * len(selection)
        + 2 * rows * law.width
        + 32 * len(draws)
    )
    report = limits.admit_dimensions(
        (law.width,),
        ((),),
        rows=max(rows, 1),
        calibration_rows=rows,
        features=law.width,
        outputs=law.width,
        extra_parameters=parameters,
        extra_working_channels=12 * law.width,
    )
    snapshots = 8 * profile.raw_limits.state_bytes + (8 * raw_limits.state_bytes if fitted else 0)
    report["estimated_working_bytes"] += snapshots + raw_bytes
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("sampled root aggregate working limit exceeded")
    return report
