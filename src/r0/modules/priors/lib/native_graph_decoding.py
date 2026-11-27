"""Bounded native construction decoding by full retained-field replay comparison."""

from dataclasses import fields as dataclass_fields

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.native_domain_prior import NativeDomainPrior
from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.native_graph_sampling import sample_native_graph
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_function_prior,
)
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph


def native_graph_from_dict(data: object, *, limits: CausalLimits) -> SampledNativeGraph:
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledNativeGraph)
        request = functional_record(record["request"], NativeGraphRequest)
        declared = CausalLimits(**functional_record(request["limits"], CausalLimits))
        if any(
            getattr(declared, f.name) > getattr(limits, f.name) for f in dataclass_fields(limits)
        ):
            raise ValueError("native construction declared limits exceed caller limits")
        admit_function_metadata(data, declared)
        request["limits"] = declared
        request["raw_limits"] = RawDrawLimits(
            **functional_record(request["raw_limits"], RawDrawLimits)
        )
        domains: list[NativeDomainPrior] = []
        for raw in functional_array(request["domains"]):
            domain = functional_record(raw, NativeDomainPrior)
            for key in (
                "kinds",
                "cardinalities",
                "emissions",
                "frequency_laws",
                "threshold_laws",
                "concentration",
                "temperature",
                "threshold_scale",
            ):
                if key in domain:
                    domain[key] = functional_array(domain[key])
            domains.append(NativeDomainPrior(**domain))
        request["domains"] = tuple(domains)
        for key in ("parents", "observed"):
            request[key] = tuple(functional_array(row) for row in functional_array(request[key]))
        request["target"] = functional_array(request["target"])
        request["function_prior"] = decode_joint_function_prior(request["function_prior"])
        root = functional_record(request["root_prior"], JointRootPrior)
        root["families"] = functional_array(root["families"])
        root["function"] = decode_joint_function_prior(root["function"])
        request["root_prior"] = JointRootPrior(**root)
        frozen = NativeGraphRequest(**request)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("native construction requires four initial streams")
        streams = tuple(
            GeneratorSnapshot(**functional_record(raw, GeneratorSnapshot)).restore(
                byte_limit=frozen.raw_limits.state_bytes
            )
            for raw in initial
        )
        graph = sample_native_graph(
            frozen,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(graph, data):
            raise ValueError("native construction archive differs from complete replay")
        return graph
    except (TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError("malformed sampled native archive") from error
