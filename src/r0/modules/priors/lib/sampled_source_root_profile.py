"""Immutable source-root law selections with exact two-stream reconstruction."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.source_root_laws import SourceRootCoordinate, SourceRootLaw
from r0.modules.priors.lib.source_root_profile_sampling import realize_root_profile
from r0.modules.priors.lib.source_root_profiles import SourceRootProfile

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class SampledSourceRootProfile:
    profile: SourceRootProfile
    law: SourceRootLaw
    model_gaussians: tuple[SourceRootCoordinate, ...]
    thresholds: tuple[float, ...]
    initial_rng: tuple[GeneratorSnapshot, ...]
    final_rng: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    source: str
    limits: CausalLimits = _DEFAULT_LIMITS
    version: str = "sampled-source-root-profile-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-source-root-profile-v1"
            or type(self.profile) is not SourceRootProfile
            or type(self.law) is not SourceRootLaw
            or self.law.width != self.profile.width
            or type(self.model_gaussians) is not tuple
            or len(self.model_gaussians) != self.profile.gaussian_width
            or any(type(v) is not SourceRootCoordinate for v in self.model_gaussians)
            or type(self.thresholds) is not tuple
            or len(self.thresholds) != (3 if self.profile.selector == "p1-code-mixed" else 0)
            or self.source != self.profile.source
            or type(self.raw) is not tuple
            or any(type(v) is not RawDrawRecord for v in self.raw)
            or any(
                type(seq) is not tuple
                or len(seq) != 2
                or any(type(v) is not GeneratorSnapshot for v in seq)
                for seq in (self.initial_rng, self.final_rng)
            )
        ):
            raise ValueError("invalid sampled source-root envelope")
        self.validate_replay(limits=self.limits)

    def validate_replay(self, *, limits: CausalLimits = _DEFAULT_LIMITS) -> None:
        self.profile.admit(limits)
        specs = self.profile.worst_draws()
        if len(self.raw) > len(specs) or sum(len(r.values) for r in self.raw) > sum(
            s.numbers for s in specs
        ):
            raise ValueError("source root raw envelope exceeds declared profile")
        rngs = tuple(
            s.restore(byte_limit=self.profile.raw_limits.state_bytes) for s in self.initial_rng
        )
        ledger = RawDrawLedger(limits=self.profile.raw_limits)
        with ledger.transactions(dict(zip(("model", "selection"), rngs, strict=True))) as (
            streams,
            capture,
        ):
            result = realize_root_profile(
                self.profile, streams["model"], streams["selection"], capture
            )
            final = tuple(
                snapshot_generator(streams[name], byte_limit=self.profile.raw_limits.state_bytes)
                for name in ("model", "selection")
            )
            if (
                not exact_frozen_equal(result, (self.law, self.model_gaussians, self.thresholds))
                or not exact_frozen_equal(capture.freeze(), self.raw)
                or not exact_frozen_equal(final, self.final_rng)
            ):
                raise ValueError("source root profile differs from raw replay")

    @property
    def source_normalization(self) -> tuple[str, ...]:
        """Required source operations, not transforms applied by this frozen law."""
        if self.profile.selector != "p1-code-mixed":
            return ("none",) * self.law.width
        return tuple(
            "batch-sample-standardization"
            if c.family == "categorical"
            else "batch-centering"
            if c.family == "zipf"
            else "none"
            for c in self.law.coordinates
        )


def sample_source_root_profile(
    profile: SourceRootProfile,
    *,
    model_rng: np.random.Generator,
    selection_rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledSourceRootProfile:
    if type(profile) is not SourceRootProfile:
        raise ValueError("source root sampling requires a frozen profile")
    profile.admit(limits)
    ledger = RawDrawLedger(limits=profile.raw_limits)
    with ledger.transactions({"model": model_rng, "selection": selection_rng}) as (
        streams,
        capture,
    ):
        initial = tuple(
            snapshot_generator(streams[name], byte_limit=profile.raw_limits.state_bytes)
            for name in ("model", "selection")
        )
        law, gaussian, thresholds = realize_root_profile(
            profile, streams["model"], streams["selection"], capture
        )
        final = tuple(
            snapshot_generator(streams[name], byte_limit=profile.raw_limits.state_bytes)
            for name in ("model", "selection")
        )
        return SampledSourceRootProfile(
            profile,
            law,
            gaussian,
            thresholds,
            initial,
            final,
            capture.freeze(),
            profile.source,
            limits,
        )
