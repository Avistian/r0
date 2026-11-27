"""Replayable source-law selection and independent normalization provenance."""

from dataclasses import dataclass, field

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.sampled_source_root_admission import admit_sampled_source_root
from r0.modules.priors.lib.sampled_source_root_profile import SampledSourceRootProfile
from r0.modules.priors.lib.source_root_base import SourceRootBase
from r0.modules.priors.lib.source_root_calibration import calibrate_source_root
from r0.modules.priors.lib.source_root_sampling import source_root_draw_specs

_DEFAULT_LIMITS = CausalLimits()
_DEFAULT_RAW_LIMITS = RawDrawLimits()


@dataclass(frozen=True, slots=True)
class SampledSourceRootBase(SourceRootBase):
    """Independent fitted adaptation, never source-style query/batch normalization."""

    profile: SampledSourceRootProfile = field(kw_only=True)
    modes: tuple[str, ...] = field(kw_only=True)
    calibration_rows: int = field(default=0, kw_only=True)
    calibration_initial: GeneratorSnapshot | None = field(default=None, kw_only=True)
    calibration_final: GeneratorSnapshot | None = field(default=None, kw_only=True)
    calibration_raw: tuple[RawDrawRecord, ...] = field(default=(), kw_only=True)
    limits: CausalLimits = field(default=CausalLimits(), kw_only=True)
    raw_limits: RawDrawLimits = field(default=RawDrawLimits(), kw_only=True)
    version: str = "sampled-source-root-base-v1"

    def __post_init__(self) -> None:
        SourceRootBase.validate_law_and_calibration(self)
        if (
            self.version != "sampled-source-root-base-v1"
            or type(self.profile) is not SampledSourceRootProfile
            or self.law != self.profile.law
            or type(self.calibration_raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.calibration_raw)
        ):
            raise ValueError("invalid sampled source root provenance")
        self.validate_replay(limits=self.limits)

    def validate_replay(self, *, limits: CausalLimits = _DEFAULT_LIMITS) -> None:
        admit_sampled_source_root(
            self.profile.profile,
            self.law,
            self.modes,
            self.calibration_rows,
            limits,
            self.raw_limits,
        )
        function_record_cost(self, limits)
        if self.calibration_rows:
            if (
                self.calibration is None
                or type(self.calibration_initial) is not GeneratorSnapshot
                or type(self.calibration_final) is not GeneratorSnapshot
                or len(self.calibration_raw) != self.width
                or sum(len(r.values) for r in self.calibration_raw)
                != self.calibration_rows * self.width
            ):
                raise ValueError("independent calibration custody geometry differs")
        elif (
            self.calibration is not None
            or self.calibration_initial is not None
            or self.calibration_final is not None
            or self.calibration_raw != ()
        ):
            raise ValueError("identity root cannot retain unused calibration custody")
        self.profile.validate_replay(limits=limits)
        if not self.calibration_rows:
            return
        assert self.calibration_initial is not None
        rng = self.calibration_initial.restore(byte_limit=self.raw_limits.state_bytes)
        ledger = RawDrawLedger(limits=self.raw_limits)
        actual = calibrate_source_root(
            self.law,
            modes=self.modes,
            rows=self.calibration_rows,
            rng=rng,
            capture=ledger,
            limits=limits,
        )
        if (
            actual != self.calibration
            or ledger.freeze() != self.calibration_raw
            or snapshot_generator(rng, byte_limit=self.raw_limits.state_bytes)
            != self.calibration_final
        ):
            raise ValueError("sampled root calibration differs from exact independent replay")


def sample_source_root_base(
    profile: SampledSourceRootProfile,
    *,
    modes: tuple[str, ...],
    calibration_rows: int = 0,
    calibration_rng: np.random.Generator | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
    raw_limits: RawDrawLimits = _DEFAULT_RAW_LIMITS,
) -> SampledSourceRootBase:
    """Freeze independent fit draws atomically; identity consumes no calibration RNG."""
    if type(profile) is not SampledSourceRootProfile:
        raise ValueError("sampled root requires complete sampled profile provenance")
    admit_sampled_source_root(
        profile.profile, profile.law, modes, calibration_rows, limits, raw_limits
    )
    function_record_cost(profile, limits)
    if not calibration_rows:
        if calibration_rng is not None:
            raise ValueError("identity root does not consume a calibration RNG")
        return SampledSourceRootBase(
            profile.law,
            profile=profile,
            modes=modes,
            limits=limits,
            raw_limits=raw_limits,
        )
    if type(calibration_rng) is not np.random.Generator:
        raise ValueError("independent root normalization requires a calibration RNG")
    ledger = RawDrawLedger(limits=raw_limits)
    ledger.admit_draws(
        calibration_rng,
        source_root_draw_specs(
            profile.law,
            calibration_rows,
            ("source-root-calibration",),
        ),
    )
    with ledger.transaction(calibration_rng) as (rng, staged):
        initial = snapshot_generator(rng, byte_limit=raw_limits.state_bytes)
        fit = calibrate_source_root(
            profile.law,
            modes=modes,
            rows=calibration_rows,
            rng=rng,
            capture=staged,
            limits=limits,
        )
        result = SampledSourceRootBase(
            profile.law,
            fit,
            profile=profile,
            modes=modes,
            calibration_rows=calibration_rows,
            calibration_initial=initial,
            calibration_final=snapshot_generator(rng, byte_limit=raw_limits.state_bytes),
            calibration_raw=staged.freeze(),
            limits=limits,
            raw_limits=raw_limits,
        )
    return result
