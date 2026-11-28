"""Source-authored registry settings are fixed by sample_joint_graph, not its archive."""

from typing import Any

from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointRegistryLimits
from r0.modules.priors.lib.joint_sampling_records import numerical_policy
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches


def validate_joint_source_contract(raw: dict[str, Any], request: JointGraphRequest) -> None:
    registry = raw["registry"]
    decisions = registry["decisions"]
    numerical, categorical = decisions["numerical"], decisions["categorical"]
    # sample_joint_graph supplies only request modes to JointSamplingContext.
    # Its registry/preference limits and root operation use the source defaults.
    expected = (
        (JointRegistryLimits(), registry["limits"]),
        (("graph",), raw["plan"]["operation"]),
        (PreferenceLimits(), decisions["limits"]),
        (PreferenceLimits(), numerical["limits"]),
        (PreferenceLimits(), categorical["limits"]),
        (request.mode, numerical["default_mode"]),
        (request.mode, categorical["default_mode"]),
        (request.global_policy, decisions["global_policy"]),
        (numerical_policy(request.global_policy), numerical["global_policy"]),
        (request.categorical_policy, categorical["policy"]),
    )
    if any(not construction_matches(source, retained) for source, retained in expected):
        raise ValueError("joint source registry settings differ from the sampler contract")
    # Every meta categorical choice creates ParameterSampler(rng) with no overrides.
    # That nested sampler deliberately retains meta/strict defaults even when the
    # outer numerical sampler uses the source quantile policy.
    for preference in categorical["preferences"]:
        hyper = preference["hyperparameters"]
        if hyper is not None and (
            hyper["default_mode"] != "meta"
            or hyper["global_policy"] != "strict"
            or not construction_matches(PreferenceLimits(), hyper["limits"])
        ):
            raise ValueError("joint source categorical hyper-scope differs from sampler contract")
