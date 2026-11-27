"""Array-free source geometry previews and construction reserves before receipt replay."""

from dataclasses import fields, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_decoding import functional_root
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.network_decoding import decode_network, decode_network_request, record
from r0.modules.priors.lib.network_hyperprior_decoding import hyperprior_request_from_dict
from r0.modules.priors.lib.network_hyperprior_records import NetworkHyperpriorRequest
from r0.modules.priors.lib.network_innovation_decoding import decode_innovation_plan
from r0.modules.priors.lib.network_innovation_plan import NetworkInnovationPlan
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_sampling_records import (
    NetworkSamplingRequest,
    network_construction_cost,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction


def values(value: Any, kind: type[Any]) -> dict[str, Any]:
    return (
        {f.name: getattr(value, f.name) for f in fields(kind)}
        if type(value) is kind
        else record(value, kind)
    )


def field(value: Any, name: str) -> Any:
    return cast(dict[str, Any], value)[name] if type(value) is dict else getattr(value, name)


def network_replay_work(network: Any, limits: CausalLimits) -> tuple[int, int]:
    version = field(network, "version")
    if version == "frozen-structural-network-v1":
        return 0, 0
    if version not in ("sampled-structural-network-v1", "sampled-hyperprior-network-v1"):
        raise ValueError("observation source network version differs")
    raw = field(network, "request")
    request = raw if type(raw) is NetworkSamplingRequest else decode_network_request(raw, limits)
    if any(getattr(request.limits, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("observation source sampling limits exceed caller limits")
    request.admit()
    numbers, draws = network_construction_cost(request)
    parameters = request.admit()["parameters"]
    work = request.admit()["estimated_working_bytes"] + 1024 * draws + 512 * numbers
    work += 2 * request.raw_limits.state_bytes
    if version == "sampled-hyperprior-network-v1":
        raw = field(network, "profile")
        profile = (
            raw
            if type(raw) is NetworkHyperpriorRequest
            else hyperprior_request_from_dict(raw, limits=limits)
        )
        if any(
            getattr(profile.template.limits, f.name) > getattr(limits, f.name)
            for f in fields(CausalLimits)
        ):
            raise ValueError("observation hyperprior limits exceed caller limits")
        profile.admit()
        maximum = replace(
            profile.template,
            widths=(profile.max_hidden_width,) * profile.max_hidden_blocks
            + profile.template.widths,
        )
        n, d = network_construction_cost(maximum)
        metadata, storage = function_record_cost(profile, limits)
        parameters = max(
            parameters,
            maximum.admit()["parameters"] + (profile.max_hidden_blocks + 3) * 256 + metadata,
        )
        work = max(
            work,
            maximum.admit()["estimated_working_bytes"]
            + 1024 * d
            + 512 * n
            + 6 * maximum.raw_limits.state_bytes
            + 16384 * (profile.max_hidden_blocks + 3)
            + 4 * storage,
        )
    return parameters, work


def root_preview(root: Any, limits: CausalLimits) -> tuple[Any, int, int]:
    """Preview root geometry and construction reserves without restoring RNG."""
    from r0.modules.priors.lib.prototype_root_base import PrototypeRootBase
    from r0.modules.priors.lib.prototype_root_construction import (
        admit_prototype_plan,
        preflight_raw,
    )
    from r0.modules.priors.lib.prototype_root_decoding import (
        frozen_prototype_root_from_dict,
        prototype_weight_law_from_dict,
    )
    from r0.modules.priors.lib.prototype_root_records import FrozenPrototypeRoot, PrototypeRootPlan
    from r0.modules.priors.lib.sampled_source_root_admission import admit_sampled_source_root
    from r0.modules.priors.lib.source_root_base import SourceRootBase, source_root_base_from_dict
    from r0.modules.priors.lib.source_root_profile_decoding import source_root_profile_from_dict
    from r0.modules.priors.lib.source_root_profiles import SourceRootProfile

    if root is None:
        return None, 0, 0
    version = field(root, "version")
    if version == "sampled-source-root-base-v1":
        raw_profile = field(field(root, "profile"), "profile")
        profile = (
            raw_profile
            if type(raw_profile) is SourceRootProfile
            else source_root_profile_from_dict(raw_profile)
        )
        authored = (
            SourceRootBase(field(root, "law"), field(root, "calibration"))
            if type(root) is not dict
            else source_root_base_from_dict(
                {
                    "law": field(root, "law"),
                    "calibration": field(root, "calibration"),
                    "version": "source-root-base-v1",
                }
            )
        )
        raw_caps = field(root, "raw_limits")
        caps = (
            raw_caps
            if type(raw_caps) is RawDrawLimits
            else RawDrawLimits(**record(raw_caps, RawDrawLimits))
        )
        report = admit_sampled_source_root(
            profile,
            authored.law,
            tuple(field(root, "modes")),
            field(root, "calibration_rows"),
            limits,
            caps,
        )
        return authored, report["parameters"], report["estimated_working_bytes"]
    if version == "prototype-root-base-v1":
        bank = field(root, "bank")
        if field(bank, "version") == "sampled-prototype-root-v1":
            raw_limits = field(bank, "limits")
            declared = (
                raw_limits
                if type(raw_limits) is CausalLimits
                else CausalLimits(**record(raw_limits, CausalLimits))
            )
            if any(
                getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)
            ):
                raise ValueError("observation prototype limits exceed caller limits")
            raw_plan = field(bank, "plan")
            base, base_parameters, replay = root_preview(field(raw_plan, "base"), limits)
            kwargs = values(raw_plan, PrototypeRootPlan)
            kwargs["base"] = base
            if type(kwargs["weights"]) is dict:
                kwargs["weights"] = prototype_weight_law_from_dict(
                    cast(dict[str, Any], kwargs["weights"])
                )
            if type(kwargs["raw_limits"]) is dict:
                kwargs["raw_limits"] = RawDrawLimits(**record(kwargs["raw_limits"], RawDrawLimits))
            plan = PrototypeRootPlan(**kwargs)
            admit_prototype_plan(plan, limits, batch=False)
            preflight_raw(plan, batch=False)
            parameters, storage = function_record_cost(plan, limits)
            rows, width = plan.count, plan.base.width
            report = limits.admit_dimensions(
                (width,),
                ((),),
                rows=rows,
                calibration_rows=0,
                features=width,
                outputs=width,
                extra_parameters=parameters + 4 * rows * width + 3 * (rows + width),
                extra_working_channels=8 * rows + 4 * width,
            )
            replay += (
                report["estimated_working_bytes"] + 4 * storage + 8 * plan.raw_limits.state_bytes
            )
            replay_parameters = base_parameters + report["parameters"]
            raw_frozen = field(bank, "root")
        else:
            replay_parameters, replay, raw_frozen = 0, 0, bank
        frozen = (
            raw_frozen
            if type(raw_frozen) is FrozenPrototypeRoot
            else frozen_prototype_root_from_dict(raw_frozen, limits=limits)
        )
        return PrototypeRootBase(frozen), replay_parameters, replay
    return (root if type(root) is not dict else functional_root(root, limits=limits)), 0, 0


def observation_source_preview(
    source: Any, limits: CausalLimits
) -> tuple[NetworkRecipe, NetworkInnovationPlan, int, int]:
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.linear import LinearRecipe
    from r0.modules.priors.lib.network_innovation_decoding import (
        network_innovation_request_from_dict,
    )
    from r0.modules.priors.lib.network_innovation_sampling_records import NetworkInnovationRequest

    noisy = field(source, "version") == "standalone-network-innovations-v1"
    base = field(source, "source") if noisy else source
    data = values(base, NetworkRecipe)
    declared = (
        data["limits"]
        if type(data["limits"]) is CausalLimits
        else CausalLimits(**record(data["limits"], CausalLimits))
    )
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("observation source declared limits exceed caller limits")
    network = data["network"]
    replay_parameters, replay = network_replay_work(network, limits)
    frozen = (
        network
        if field(network, "version") == "frozen-structural-network-v1"
        else field(network, "function")
    )
    function = frozen if type(frozen) is FrozenNetworkFunction else decode_network(frozen)
    root, root_parameters, root_work = root_preview(data["root"], limits)
    replay += root_work
    replay_parameters += root_parameters
    task = data["task"] if type(data["task"]) is LinearRecipe else recipe_from_dict(data["task"])
    if type(task) is not LinearRecipe:
        raise ValueError("observation source task controls differ")
    preview = NetworkRecipe(task, function, root, declared)
    if not noisy:
        return (
            preview,
            NetworkInnovationPlan((None,) * len(function.blocks)),
            replay_parameters,
            replay,
        )
    raw_plan = field(source, "plan")
    plan = raw_plan if type(raw_plan) is NetworkInnovationPlan else decode_innovation_plan(raw_plan)
    sampling = field(source, "sampling")
    if sampling is not None:
        raw = field(sampling, "request")
        request = (
            raw
            if type(raw) is NetworkInnovationRequest
            else network_innovation_request_from_dict(raw, limits=limits)
        )
        count = (
            2
            * sum(
                b.output_width
                for i, b in enumerate(function.blocks)
                if not (request.exempt_first and i == 0)
            )
            + function.input_width
        )
        metadata, storage = function_record_cost(request, limits)
        replay_parameters += metadata + 512 * count
        replay += 32768 * count + 4 * request.raw_limits.state_bytes + 4 * storage
    return preview, plan, replay_parameters, replay
