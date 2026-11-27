"""Typed policy task, comparison and trace dispatch retains the original base envelope."""

import json
from dataclasses import asdict, replace
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.causal_intervention_inspection import intervention_steps
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.intervention_policy_validation import validate_policy_task
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.typed_policy_decoding import stored_base_limits, typed_policy_from_dict
from r0.modules.priors.lib.typed_policy_records import SampledTypedInterventionPolicy


def generate_typed_policy_task(
    recipe: SampledTypedInterventionPolicy, *, seed: int, task_id: int, device: str = "cpu"
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
        "locations": [asdict(a) for a in recipe.request.locations],
        "bounds": [asdict(b) for b in recipe.bounds],
        "parameter_laws": [asdict(law) for law in recipe.request.parameter_laws],
        "bounds_lifecycle": recipe.request.bounds_lifecycle if recipe.bounds else "not-used",
        "reference_seed": recipe.reference_seed if recipe.bounds else None,
        "policy_raw_draw_count": len(recipe.raw),
        "reference_raw_draw_count": len(recipe.reference_raw),
        "notice": (
            "Explicit r0 typed policy: categorical choices follow declared vocabulary "
            "probabilities, not source marginal frequencies. Original construction/fit "
            "custody remains retained. Native value, numeric readout/state and converter "
            "capture/representation locations are distinct."
        ),
    }
    return replace(
        task,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def generate_typed_policy_pair(recipe: SampledTypedInterventionPolicy, *, seed: int, task_id: int):
    from r0.modules.priors.lib.causal_intervention_generation import compare_causal_pair
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.pairs import TaskPair

    before = generate_task(recipe.request.base, seed=seed, task_id=task_id)
    after = generate_typed_policy_task(recipe, seed=seed, task_id=task_id)
    return TaskPair(
        before, after, encode(compare_causal_pair(before, after, limits=recipe.request.limits))
    )


def compare_typed_policy_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None
):
    from r0.modules.priors.lib.causal_intervention_generation import compare_causal_pair

    raw = json.loads(after.recipe_json)
    recipe = typed_policy_from_dict(raw["recipe"], limits=limits or CausalLimits())
    replay = generate_typed_policy_task(recipe, seed=raw["seed"], task_id=raw["task_id"])
    validate_policy_task(after, replay)
    plain = generate_intervened(recipe.runtime, seed=raw["seed"], task_id=raw["task_id"])
    return {
        **compare_causal_pair(
            before, plain, limits=stored_base_limits(asdict(recipe.request.base))
        ),
        "sampled_policy": json.loads(after.mechanism_json)["sampled_policy"],
        "policy_source_task_id": after.source_task_id,
    }


def typed_policy_steps(task: TrainingTask, indices: Any, add: Any, *, limits: CausalLimits | None):
    record = json.loads(task.recipe_json)
    if any(type(record[k]) is not int or not 0 <= record[k] < 2**63 for k in ("seed", "task_id")):
        raise ValueError("typed policy trace requires exact task identities")
    recipe = typed_policy_from_dict(record["recipe"], limits=limits or CausalLimits())
    replay = generate_typed_policy_task(recipe, seed=record["seed"], task_id=record["task_id"])
    validate_policy_task(task, replay)
    plain = generate_intervened(recipe.runtime, seed=record["seed"], task_id=record["task_id"])
    selection_draws: list[str] = []
    value_draws: list[str] = []
    for i, draw in enumerate(recipe.raw):
        if draw.role == "calibration-seed":
            continue  # Retained exactly as a decimal string with the reference bounds.
        identifier = f"policy-draw-{i}"
        add(
            identifier,
            f"Retained policy draw: {draw.role}",
            [],
            draw.distribution,
            torch.tensor(draw.values, dtype=torch.float64).expand(len(indices), -1),
            {
                "owner": draw.owner,
                "stream": draw.stream,
                "sequence": draw.sequence,
                "shape": draw.shape,
                "parameters": draw.parameters,
                "generator": draw.generator,
                "state_before": draw.state_before,
                "state_after": draw.state_after,
            },
        )
        (selection_draws if draw.role == "eligible-permutation" else value_draws).append(identifier)
    if recipe.bounds:
        values = tuple(x for bound in recipe.bounds for x in (*bound.minimum, *bound.maximum))
        add(
            "policy-reference-bounds",
            "Frozen independent/base-population bounds",
            [],
            "complete retained reference extrema, not query fitting",
            torch.tensor(values, dtype=torch.float64).expand(len(indices), -1),
            {
                "bounds": [asdict(b) for b in recipe.bounds],
                "reference_seed": str(recipe.reference_seed),
                "reference_task_id": str(recipe.request.reference_task_id),
                "rows": recipe.request.reference_rows,
                "raw_draw_count": len(recipe.reference_raw),
                "lifecycle": recipe.request.bounds_lifecycle,
            },
        )
        value_draws.append("policy-reference-bounds")
    add(
        "policy-selection",
        "Typed sampled intervention targets",
        selection_draws,
        "uniform permutation prefix over nodes with eligible non-label addresses",
        torch.tensor(recipe.selected, dtype=torch.float64).expand(len(indices), -1),
        {"profile": recipe.request.profile, "eligible": recipe.request.eligible},
    )

    add(
        "policy-realization",
        "Retained typed intervention realization",
        ["policy-selection", *value_draws],
        "one frozen categorical choice, numerical clamp or parameter realization per environment",
        torch.tensor(recipe.selected, dtype=torch.float64).expand(len(indices), -1),
        {
            "changes": [asdict(c) for c in recipe.runtime.interventions],
            "bounds": [asdict(b) for b in recipe.bounds],
            "locations": [asdict(a) for a in recipe.request.locations],
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
        parts = identifier.split("-", 2)
        depends = False
        if len(parts) == 3 and parts[0] == "intervention" and parts[1].isdigit():
            node, role = int(parts[1]), parts[2]
            if recipe.request.kind == "soft":
                depends = node in recipe.selected and role == "function"
            else:
                for address in recipe.request.locations:
                    if address.node != node or node not in recipe.selected:
                        continue
                    if address.kind == "numerical":
                        depends |= role == "state" and address.location != "readout"
                        depends |= role == "readout" and address.location != "state"
                    elif address.kind == "native-variable":
                        depends |= role in ("state", "value", "forced_probability")
                        depends |= role in (
                            f"value:{address.variable}",
                            f"forced-score:{address.variable}",
                            f"forced-probability:{address.variable}",
                        )
                    else:
                        depends |= role in (
                            f"capture:{address.capture}",
                            f"probabilities:{address.capture}",
                        )
                        depends |= address.location == "representation_and_capture" and role in (
                            "state",
                            f"representation:{address.capture}",
                        )
        if depends:
            parents = [*parents, "policy-realization"]
        add(identifier, title, parents, operation, values, metadata)

    trace = intervention_steps(plain, indices, policy_add, limits=limits)
    return {
        **trace,
        "episode_id": task.episode_id,
        "sampled_policy": json.loads(task.mechanism_json)["sampled_policy"],
    }
