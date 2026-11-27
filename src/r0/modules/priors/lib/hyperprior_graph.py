"""Versioned topology construction attached to an explicit shared-template task."""

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.hyperprior_graph_request import HyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.sampled_topology_hyperprior import (
    SampledTopologyHyperprior,
    sample_topology_hyperprior,
)
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class HyperpriorGraph:
    request: HyperpriorGraphRequest
    topology: SampledTopologyHyperprior
    version: str = "sampled-hyperprior-graph-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-hyperprior-graph-v1"
            or type(self.request) is not HyperpriorGraphRequest
            or type(self.topology) is not SampledTopologyHyperprior
        ):
            raise ValueError("hyperprior graph requires its complete construction envelope")
        self.request.admit(retained=self.topology)
        if (
            not exact_frozen_equal(self.request.topology, self.topology.request)
            or self.request.limits != self.topology.limits
            or self.request.raw_limits != self.topology.raw_limits
        ):
            raise ValueError("topology graph construction differs from its request")
        self.topology.validate_structure()
        self.runtime.admit()

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    @property
    def runtime(self) -> FunctionalCausalRecipe:
        return self.request.runtime(self.topology.parents)

    def validate(self) -> None:
        self.request.admit(retained=self.topology)
        self.topology.replay()

    def with_task(self, task: LinearRecipe) -> "HyperpriorGraph":
        """Task controls may change; construction is retained, never silently resampled."""
        return replace(self, request=replace(self.request, task=task))


def sample_hyperprior_graph(request: HyperpriorGraphRequest, *, rng: np.random.Generator):
    if type(request) is not HyperpriorGraphRequest:
        raise ValueError("topology graph sampler requires a frozen request")
    request.admit()
    with raw_transactions({"topology": rng}, None) as (staged, _):
        topology = sample_topology_hyperprior(
            request.topology,
            rng=staged["topology"],
            limits=request.limits,
            raw_limits=request.raw_limits,
        )
        return HyperpriorGraph(request, topology)


def generate_hyperprior_graph(
    recipe: HyperpriorGraph, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal

    recipe.validate()
    task = generate_functional_causal(recipe.runtime, seed=seed, task_id=task_id, device=device)
    record, card = json.loads(task.recipe_json), json.loads(task.mechanism_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    card.update(hyperprior_graph_card(recipe))
    config = asdict(recipe)
    config["request"]["task"] = {
        k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")
    }
    identity = record_fingerprint((config, seed, task_id))
    return replace(
        task,
        source_task_id=identity,
        episode_id=record_fingerprint((identity, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def hyperprior_graph_card(recipe: HyperpriorGraph) -> dict[str, Any]:
    """Construction facts compared separately from historical environment provenance."""
    return dict(
        version=recipe.version,
        family="hyperprior_graph",
        topology=asdict(recipe.topology.topology),
        topology_components=recipe.topology.components,
        construction=recipe.request.mechanism_sharing,
        topology_layout=recipe.request.layout,
        topology_receipt_fingerprint=record_fingerprint(asdict(recipe.topology)),
        resource_admission=recipe.request.admit(retained=recipe.topology),
    )
