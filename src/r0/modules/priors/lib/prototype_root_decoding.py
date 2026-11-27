"""Strict bounded prototype root records; batch references never decode as frozen banks."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_root
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.prototype_root_construction import (
    PrototypeBatchReference,
    SampledPrototypeRoot,
    admit_prototype_plan,
    bank_draw_specs,
)
from r0.modules.priors.lib.prototype_root_emission import PrototypeRootDraw
from r0.modules.priors.lib.prototype_root_records import (
    FrozenPrototypeRoot,
    PrototypeRootPlan,
    admit_prototype_work,
)
from r0.modules.priors.lib.prototype_weight_law import PrototypeWeightLaw
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

_DEFAULT_LIMITS = CausalLimits()


def prototype_weight_law_from_dict(data: object) -> PrototypeWeightLaw:
    values = fitted_object(data, PrototypeWeightLaw)
    values["probabilities"] = fitted_array(values["probabilities"], 4096)
    return PrototypeWeightLaw(**values)


def prototype_root_plan_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PrototypeRootPlan:
    admit_function_metadata(data, limits)
    values = fitted_object(data, PrototypeRootPlan)
    values["base"] = functional_root(values["base"])
    values["weights"] = prototype_weight_law_from_dict(values["weights"])
    values["raw_limits"] = RawDrawLimits(**fitted_object(values["raw_limits"], RawDrawLimits))
    plan = PrototypeRootPlan(**values)
    admit_prototype_plan(plan, limits, batch=False)
    return plan


def frozen_prototype_root_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FrozenPrototypeRoot:
    admit_function_metadata(data, limits)
    values = fitted_object(data, FrozenPrototypeRoot)
    raw = fitted_array(values["prototypes"], 4096)
    if not raw:
        raise ValueError("prototype bank must be nonempty")
    width = len(fitted_array(raw[0], 1024))
    admit_prototype_work(len(raw), width, 1, limits)
    values["prototypes"] = tuple(fitted_array(row, width) for row in raw)
    values["weights"] = prototype_weight_law_from_dict(values["weights"])
    return FrozenPrototypeRoot(**values)


def _snapshot(data: object, bound: int) -> GeneratorSnapshot:
    values = fitted_object(data, GeneratorSnapshot)
    if type(values["payload"]) is not str or len(values["payload"]) > bound:
        raise ValueError("prototype snapshot exceeds its declared byte bound")
    return GeneratorSnapshot(**values)


def decode_prototype_record(data: object, *, batch: bool, limits: CausalLimits) -> Any:
    admit_function_metadata(data, limits)
    kind = PrototypeBatchReference if batch else SampledPrototypeRoot
    values = fitted_object(data, kind)
    declared = CausalLimits(**fitted_object(values["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(limits)):
        raise ValueError("prototype declared limits exceed caller limits")
    plan = prototype_root_plan_from_dict(values["plan"], limits=declared)
    admit_prototype_plan(plan, declared, batch=batch)
    values["plan"], values["limits"] = plan, declared
    root = fitted_object(values["root"], FrozenPrototypeRoot)
    matrix = fitted_array(root["prototypes"], plan.count)
    if len(matrix) != plan.count:
        raise ValueError("prototype retained count differs")
    root["prototypes"] = tuple(fitted_array(row, plan.base.width) for row in matrix)
    root["weights"] = prototype_weight_law_from_dict(root["weights"])
    values["root"] = FrozenPrototypeRoot(**root)
    specs = bank_draw_specs(plan)
    if batch:
        specs = (
            *specs,
            plan.weights.draw_spec(plan.count, plan.reference_rows, ("prototype-emission",)),
        )
        draw = fitted_object(values["draw"], PrototypeRootDraw)
        for name, columns in (
            ("values", plan.base.width),
            ("weights", plan.count),
            ("counts", plan.count),
        ):
            if name == "counts" and draw[name] is None:
                continue
            draw[name] = tuple(
                fitted_array(row, columns) for row in fitted_array(draw[name], plan.reference_rows)
            )
        values["draw"] = PrototypeRootDraw(**draw)
    raw = fitted_array(values["raw"], len(specs))
    if len(raw) != len(specs):
        raise ValueError("prototype raw record count differs")
    decoded: list[Any] = []
    for record, spec in zip(raw, specs, strict=True):
        if type(record) is not dict:
            raise ValueError("prototype raw record must be an object")
        item = cast(dict[str, Any], record)
        fitted_array(item.get("values"), spec.numbers)
        # Parameters are bounded by the declared law before nested tuple conversion.
        for pair in fitted_array(item.get("parameters"), 2):
            entry = fitted_array(pair, 2)
            if len(entry) != 2:
                raise ValueError("prototype raw parameter fields differ")
            if type(entry[1]) in (tuple, list):
                fitted_array(entry[1], max(plan.count, 1024))
        decoded.append(decode_fitted_raw_draw(item))
    values["raw"] = tuple(decoded)
    for name in ("initial", "final"):
        values[name] = (
            tuple(_snapshot(s, plan.raw_limits.state_bytes) for s in fitted_array(values[name], 2))
            if batch
            else _snapshot(values[name], plan.raw_limits.state_bytes)
        )
    return kind(**values)


def sampled_prototype_root_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledPrototypeRoot:
    return decode_prototype_record(data, batch=False, limits=limits)


def prototype_batch_reference_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PrototypeBatchReference:
    return decode_prototype_record(data, batch=True, limits=limits)
