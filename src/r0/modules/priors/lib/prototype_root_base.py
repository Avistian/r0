"""Frozen-bank graph roots; batch-reference populations are deliberately separate."""

from dataclasses import dataclass, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.prototype_root_construction import (
    SampledPrototypeRoot,
    admit_prototype_plan,
)
from r0.modules.priors.lib.prototype_root_records import FrozenPrototypeRoot, admit_prototype_work
from r0.modules.priors.lib.sampled_source_root_admission import admit_sampled_source_root
from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class PrototypeRootBase:
    bank: FrozenPrototypeRoot | SampledPrototypeRoot
    version: str = "prototype-root-base-v1"

    def __post_init__(self) -> None:
        if self.version != "prototype-root-base-v1" or type(self.bank) not in (
            FrozenPrototypeRoot,
            SampledPrototypeRoot,
        ):
            raise ValueError("prototype graph root requires an independent frozen bank")

    @property
    def frozen(self) -> FrozenPrototypeRoot:
        return self.bank.root if isinstance(self.bank, SampledPrototypeRoot) else self.bank

    @property
    def width(self) -> int:
        return self.frozen.width

    def admit(self, rows: int, limits: CausalLimits) -> tuple[int, int]:
        bank = self.frozen
        admit_prototype_work(bank.count, bank.width, rows, limits)
        replay_rows = 0
        if isinstance(self.bank, SampledPrototypeRoot):
            admit_prototype_plan(self.bank.plan, limits, batch=False)
            replay_rows = bank.count
        # Conservative retained Python row weights/counts, dense values and replay buffers.
        numbers = 4 * bank.count * bank.width + 3 * rows * (bank.count + bank.width)
        working = 128 * numbers + 8 * replay_rows * (8 * bank.count + 4 * bank.width)
        if isinstance(self.bank, SampledPrototypeRoot) and isinstance(
            self.bank.plan.base, SampledSourceRootBase
        ):
            source = self.bank.plan.base
            report = admit_sampled_source_root(
                source.profile.profile,
                source.law,
                source.modes,
                source.calibration_rows,
                limits,
                source.raw_limits,
            )
            numbers += report["parameters"]
            working += report["estimated_working_bytes"]
        if numbers > limits.parameters or working > limits.working_bytes:
            raise ValueError("prototype graph root workspace limit exceeded")
        return numbers, working


def prototype_root_base_from_dict(
    data: dict[str, Any], *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PrototypeRootBase:
    from r0.modules.priors.lib.prototype_root_decoding import (
        frozen_prototype_root_from_dict,
        sampled_prototype_root_from_dict,
    )

    admit_function_metadata(data, limits)
    if type(data) is not dict or set(data) != {f.name for f in fields(PrototypeRootBase)}:
        raise ValueError("prototype graph root fields differ")
    bank = data["bank"]
    if type(bank) is not dict:
        raise ValueError("prototype graph bank must be an object")
    bank = cast(dict[str, Any], bank)
    frozen = (
        sampled_prototype_root_from_dict(bank, limits=limits)
        if bank.get("version") == "sampled-prototype-root-v1"
        else frozen_prototype_root_from_dict(bank, limits=limits)
    )
    return PrototypeRootBase(frozen, data["version"])
