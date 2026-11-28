"""Observation task envelope retains source and selection custody; runtime is derived."""

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_observation_request import observation_source
from r0.modules.priors.lib.network_observation_sampling import SampledNetworkObservations
from r0.modules.priors.lib.observation_envelope_admission import admit_observation_envelope
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class NetworkObservationRecipe:
    task: LinearRecipe
    observations: SampledNetworkObservations
    version: str = "network-observation-recipe-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-observation-recipe-v1"
            or type(self.task) is not LinearRecipe
            or type(self.observations) is not SampledNetworkObservations
        ):
            raise ValueError("observation task requires its complete frozen selection receipt")
        self.validate()

    def validate(self) -> None:
        from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe
        from r0.modules.priors.lib.network_sampling_records import SampledNetwork
        from r0.modules.priors.lib.prototype_root_base import PrototypeRootBase
        from r0.modules.priors.lib.prototype_root_construction import SampledPrototypeRoot
        from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase

        self.admit()
        self.observations.validate_replay()
        source, _ = observation_source(self.observations.request)
        if isinstance(source.network, SampledNetwork):
            source.network.validate_replay()
        if isinstance(source.root, SampledSourceRootBase):
            source.root.validate_replay(limits=self.limits)
        if isinstance(source.root, PrototypeRootBase) and isinstance(
            source.root.bank, SampledPrototypeRoot
        ):
            source.root.bank.validate_replay(limits=self.limits)
        requested = self.observations.request.source
        if isinstance(requested, NetworkInnovationRecipe) and requested.sampling is not None:
            requested.sampling.validate_replay(source)

    @property
    def limits(self) -> CausalLimits:
        return observation_source(self.observations.request)[0].limits

    @property
    def runtime(self) -> FunctionalCausalRecipe:
        self.admit()
        return self.observations.runtime(self.task)

    def admit(self, *, evaluation_rows: int | None = None) -> dict[str, int]:
        return admit_observation_envelope(
            self.task, self.observations, self.limits, evaluation_rows=evaluation_rows
        )

    def selection_card(self, runtime: FunctionalCausalRecipe) -> dict[str, Any]:
        receipt = self.observations
        original = tuple(
            (receipt.catalog[i].node, receipt.catalog[i].channel) for i in receipt.targets
        )
        return {
            "profile": receipt.request.profile,
            "selection": receipt.request.selection,
            "target_mode": receipt.request.target,
            "feature_order": receipt.request.feature_order,
            "catalog": [asdict(c) for c in receipt.catalog],
            "features": receipt.features,
            "targets": receipt.targets,
            "original_target_refs": original,
            "observed_refs": runtime.observed,
            "target_refs": runtime.target,
            "target_feature_overlap": receipt.target_feature_overlap,
            "target_aliases": [
                {"original": before, "alias": after}
                for before, after in zip(original, runtime.target, strict=True)
                if before != after
            ],
            "source_receipt": "complete source, selection raw draws and snapshots in recipe",
            "overlap_semantics": "source-selected target leakage retained without filtering",
        }


def generate_network_observations(
    recipe: NetworkObservationRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
) -> TrainingTask:
    from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal

    recipe.validate()
    runtime = recipe.runtime
    task = generate_functional_causal(runtime, seed=seed, task_id=task_id, device=device)
    record = json.loads(task.recipe_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    card = json.loads(task.mechanism_json)
    card.update(
        version=recipe.version,
        family="observed_network",
        construction="source-qualified-network-observation-selection-v1",
        sampling_provenance="retained-network-source-and-observation-raw-custody-v1",
        observation_selection=recipe.selection_card(runtime),
        resource_admission=recipe.admit(),
    )
    if recipe.observations.target_feature_overlap:
        card["flags"] = [*card["flags"], "source_target_feature_overlap"]
    config = asdict(recipe)
    config["task"] = {
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
