"""Standalone sampled-function tasks, distinct from a sampled causal graph prior."""

import json
from dataclasses import asdict, dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_hyperprior_records import SampledHyperpriorNetwork
from r0.modules.priors.lib.network_sampling_records import SampledNetwork
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase
from r0.modules.priors.lib.source_root_base import SourceRootBase
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class NetworkRecipe:
    task: LinearRecipe
    network: FrozenNetworkFunction | SampledNetwork
    root: JointRootBase | SourceRootBase | None = None
    limits: CausalLimits = CausalLimits()
    version: str = "standalone-network-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "standalone-network-v1"
            or type(self.task) is not LinearRecipe
            or type(self.network)
            not in (FrozenNetworkFunction, SampledNetwork, SampledHyperpriorNetwork)
            or type(self.limits) is not CausalLimits
        ):
            raise ValueError(
                "standalone network requires its frozen function/construction and task controls"
            )
        function = self.function
        from r0.modules.priors.lib.frozen_function_evaluation import admit_frozen_function

        admit_frozen_function(function, self.task.rows, self.limits)
        expected = self.task.classes if self.task.task_type == "multiclass" else 1
        if function.input_width != self.task.features or function.output_width != expected:
            raise ValueError("standalone network input/output widths differ from task controls")
        if self.root is not None and (
            type(self.root) not in (JointRootBase, SourceRootBase, SampledSourceRootBase)
            or self.root.width != function.input_width
        ):
            raise ValueError("standalone network root must match feature width")
        self.runtime.admit()

    @property
    def function(self) -> FrozenNetworkFunction:
        return self.network.function if isinstance(self.network, SampledNetwork) else self.network

    @property
    def runtime(self) -> FunctionalCausalRecipe:
        width = self.function.input_width
        identity = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
        return FunctionalCausalRecipe(
            self.task,
            (
                FunctionalCausalNode(
                    (),
                    FrozenLinearFunction(identity),
                    self.root or JointRootBase("gaussian", width),
                    version="functional-causal-node-v2"
                    if isinstance(self.root, SourceRootBase)
                    else "functional-causal-node-v1",
                ),
                FunctionalCausalNode((0,), self.function),
            ),
            tuple((0, j) for j in range(width)),
            tuple((1, j) for j in range(self.function.output_width)),
            self.limits,
        )


def generate_network(
    recipe: NetworkRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if isinstance(recipe.network, SampledNetwork):
        recipe.network.validate_replay()
    task = generate_functional_causal(recipe.runtime, seed=seed, task_id=task_id, device=device)
    record = json.loads(task.recipe_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    card = json.loads(task.mechanism_json)
    card.update(
        family="standalone_network",
        architecture="network block DAG; input state is 0",
        dropout="frozen structural masks; no per-row mask resampling",
        source="r0 architecture extension; source mask profile identified in retained request",
    )
    config = asdict(recipe)
    config["task"] = {
        k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")
    }
    source = record_fingerprint((config, seed, task_id))
    return replace(
        task,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def network_recipe_from_dict(data: object, *, limits: CausalLimits) -> NetworkRecipe:
    from dataclasses import fields as dataclass_fields

    from r0.modules.priors.lib.function_metadata import admit_function_metadata
    from r0.modules.priors.lib.functional_causal_decoding import (
        functional_root as root_base_from_dict,
    )
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.network_decoding import (
        decode_network,
        decode_sampled_network,
        record,
    )

    admit_function_metadata(data, limits)
    try:
        fields = record(data, NetworkRecipe)
        declared = CausalLimits(**record(fields["limits"], CausalLimits))
        if any(
            getattr(declared, f.name) > getattr(limits, f.name) for f in dataclass_fields(limits)
        ):
            raise ValueError("network declared limits exceed caller limits")
        fields["limits"] = declared
        fields["task"] = recipe_from_dict(fields["task"])
        net = fields["network"]
        from r0.modules.priors.lib.network_hyperprior_decoding import decode_hyperprior_network

        fields["network"] = (
            decode_hyperprior_network(net, declared)
            if net.get("version") == "sampled-hyperprior-network-v1"
            else decode_sampled_network(net, declared)
            if net.get("version") == "sampled-structural-network-v1"
            else decode_network(net)
        )
        fields["root"] = (
            root_base_from_dict(fields["root"], limits=declared)
            if fields["root"] is not None
            else None
        )
        return NetworkRecipe(**fields)
    except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed standalone network archive") from error
