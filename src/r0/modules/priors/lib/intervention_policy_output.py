"""Task and trace envelopes preserve sampled policy custody instead of discarding it."""

import json
from dataclasses import asdict, replace
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.causal_intervention_inspection import intervention_steps
from r0.modules.priors.lib.causal_intervention_records import HardIntervention
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.intervention_policy_decoding import intervention_policy_from_dict
from r0.modules.priors.lib.intervention_policy_records import SampledInterventionPolicy
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def generate_policy_task(
    recipe: SampledInterventionPolicy, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    recipe.validate_replay()
    task = generate_intervened(recipe.runtime, seed=seed, task_id=task_id, device=device)
    record = json.loads(task.recipe_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    source = record_fingerprint((asdict(recipe), seed, task_id))
    card = json.loads(task.mechanism_json)
    card["sampled_policy"] = {
        "profile": recipe.request.profile,
        "selected": recipe.selected,
        "bounds_lifecycle": recipe.request.bounds_lifecycle
        if recipe.request.kind == "hard"
        else "not-used-soft",
        "reference_seed": recipe.reference_seed if recipe.request.kind == "hard" else None,
        "raw_draw_count": len(recipe.raw) + len(recipe.reference_raw),
        "policy_raw_draw_count": len(recipe.raw),
        "reference_raw_draw_count": len(recipe.reference_raw),
        "parameter_laws": [asdict(x) for x in recipe.request.parameter_laws],
        "notice": (
            "Source selection; explicit r0 parameter laws and declared bounds lifecycle. "
            "Fresh population and matched comparisons are distinct."
        ),
    }
    return replace(
        task,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def policy_steps(
    task: TrainingTask, indices: Any, add: Any, *, limits: CausalLimits | None
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    if any(type(record[k]) is not int or not 0 <= record[k] < 2**63 for k in ("seed", "task_id")):
        raise ValueError("policy trace requires exact task identity")
    recipe = intervention_policy_from_dict(record["recipe"], limits=limits or CausalLimits())
    replay = generate_policy_task(recipe, seed=record["seed"], task_id=record["task_id"])
    from r0.modules.priors.lib.intervention_policy_validation import validate_policy_task

    validate_policy_task(task, replay)
    plain = generate_intervened(recipe.runtime, seed=record["seed"], task_id=record["task_id"])
    add(
        "policy-selection",
        "Sampled intervention targets",
        [],
        "uniform permutation prefix over declared eligible non-label nodes",
        torch.tensor(recipe.selected, dtype=torch.float64).expand(len(indices), -1),
        {"eligible": recipe.request.eligible, "profile": recipe.request.profile},
    )
    for change in recipe.runtime.interventions:
        values = change.values if isinstance(change, HardIntervention) else (float(change.node),)
        add(
            f"policy-node-{change.node}",
            "Sampled intervention parameters",
            ["policy-selection"],
            "one retained parameter realization per environment",
            torch.tensor(values, dtype=torch.float64).expand(len(indices), -1),
            {
                "change": asdict(change),
                "bounds": [asdict(b) for b in recipe.bounds if b.node == change.node],
                "bounds_lifecycle": recipe.request.bounds_lifecycle
                if recipe.request.kind == "hard"
                else "not-used-soft",
            },
        )

    def policy_add(
        identifier: str,
        title: str,
        parents: list[str],
        operation: str,
        values: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        parts = identifier.split("-")
        if (
            len(parts) >= 3
            and parts[0] == "intervention"
            and parts[1].isdigit()
            and int(parts[1]) in recipe.selected
            and parts[2] in ("function", "state", "readout")
        ):
            parents = [*parents, f"policy-node-{parts[1]}"]
        add(identifier, title, parents, operation, values, metadata)

    trace = intervention_steps(plain, indices, policy_add, limits=limits)
    return {
        **trace,
        "episode_id": task.episode_id,
        "sampled_policy": json.loads(task.mechanism_json)["sampled_policy"],
    }


def generate_policy_pair(recipe: SampledInterventionPolicy, *, seed: int, task_id: int):
    from r0.modules.priors.lib.causal_intervention_generation import compare_causal_pair
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.pairs import TaskPair

    before = generate_task(recipe.request.base, seed=seed, task_id=task_id)
    after = generate_policy_task(recipe, seed=seed, task_id=task_id)
    return TaskPair(
        before, after, encode(compare_causal_pair(before, after, limits=recipe.request.limits))
    )


def compare_policy_pair(before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None):
    from r0.modules.priors.lib.causal_intervention_generation import compare_causal_pair
    from r0.modules.priors.lib.intervention_policy_validation import validate_policy_task

    record = json.loads(after.recipe_json)
    recipe = intervention_policy_from_dict(record["recipe"], limits=limits or CausalLimits())
    replay = generate_policy_task(recipe, seed=record["seed"], task_id=record["task_id"])
    validate_policy_task(after, replay)
    plain = generate_intervened(recipe.runtime, seed=record["seed"], task_id=record["task_id"])
    comparison = compare_causal_pair(before, plain, limits=limits)
    return {
        **comparison,
        "sampled_policy": json.loads(after.mechanism_json)["sampled_policy"],
        "policy_source_task_id": after.source_task_id,
    }
