"""Retained root replay/query costs used by forest whole-task admission, without replay."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_source_root_admission import admit_sampled_source_root
from r0.modules.priors.lib.source_root_laws import source_root_from_dict
from r0.modules.priors.lib.source_root_profile_decoding import source_root_profile_from_dict


def recursive_root_budget(
    data: Any, rows: int, width: int, limits: CausalLimits
) -> tuple[int, int]:
    if data is None:
        return 0, 0
    if type(data) is not dict:
        raise ValueError("recursive graph root must be a record")
    raw = cast(dict[str, Any], data)
    if raw.get("version") == "sampled-source-root-base-v1":
        profile = source_root_profile_from_dict(raw["profile"]["profile"])
        law = source_root_from_dict(raw["law"])
        report = admit_sampled_source_root(
            profile,
            law,
            tuple(raw["modes"]),
            raw["calibration_rows"],
            limits,
            RawDrawLimits(**raw["raw_limits"]),
        )
        return report["parameters"], report["estimated_working_bytes"]
    if raw.get("version") == "prototype-root-base-v1":
        bank = raw["bank"]
        if type(bank) is not dict:
            raise ValueError("recursive prototype root bank must be a record")
        bank = cast(dict[str, Any], bank)
        sampled = bank.get("version") == "sampled-prototype-root-v1"
        frozen = bank["root"] if sampled else bank
        if type(frozen) is not dict:
            raise ValueError("recursive frozen prototype root must be a record")
        prototypes = cast(dict[str, Any], frozen)["prototypes"]
        if type(prototypes) not in (list, tuple):
            raise ValueError("recursive frozen prototype matrix must be an array")
        count = len(cast(list[Any] | tuple[Any, ...], prototypes))
        numbers = 4 * count * width + 3 * rows * (count + width)
        working = 128 * numbers + (8 * count * (8 * count + 4 * width) if sampled else 0)
        if sampled:
            nested_numbers, nested_work = recursive_root_budget(
                bank["plan"]["base"], count, width, limits
            )
            numbers += nested_numbers
            working += nested_work
        if numbers > limits.parameters or working > limits.working_bytes:
            raise ValueError("recursive prototype root workspace limit exceeded")
        return numbers, working
    return 0, 0


def recursive_root_width(data: Any) -> int:
    if type(data) is not dict:
        raise ValueError("recursive graph root must be a record")
    raw = cast(dict[str, Any], data)
    if raw.get("version") in ("source-root-base-v1", "sampled-source-root-base-v1"):
        width = len(raw["law"]["coordinates"])
    elif raw.get("version") == "prototype-root-base-v1":
        bank = raw["bank"]
        frozen = bank["root"] if bank["version"] == "sampled-prototype-root-v1" else bank
        width = len(frozen["prototypes"][0])
    else:
        width = raw["width"]
    if type(width) is not int or width < 1:
        raise ValueError("recursive root width must be positive integer geometry")
    return width
