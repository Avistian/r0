"""Replay must reject contradictory task semantics on either side before persistence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules import priors as p
from r0.modules.generation import save_causal_pair


def pair() -> p.TaskPair:
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    base = p.FunctionalCausalRecipe(
        task,
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    return p.generate_causal_pair(
        base, interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=61, task_id=11
    )


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize(
    "field",
    [
        "task_type",
        "source_task_id",
        "episode_id",
        "classes",
        "feature_schema",
        "mechanism_json",
        "recipe_json",
        "execution_device",
    ],
)
def test_contradictory_task_metadata_rejects_before_persistence(
    side: str, field: str, tmp_path: Path
) -> None:
    original = pair()
    task = getattr(original, side)
    recipe = json.loads(task.recipe_json)
    if field == "execution_device":
        recipe["execution"]["device"] = "cuda"
    else:
        recipe["execution"]["implementation"] = "invented-implementation-v1"
    changed = replace(
        task,
        **{
            ("recipe_json" if field == "execution_device" else field): {
                "task_type": "binary",
                "source_task_id": "invented-source",
                "episode_id": "invented-episode",
                "classes": ("invented-class",),
                "feature_schema": (p.TrainingFeature("continuous"),),
                "mechanism_json": "{}",
                "recipe_json": json.dumps(recipe),
                "execution_device": json.dumps(recipe),
            }[field]
        },
    )
    tampered = replace(original, **{side: changed})
    with pytest.raises(ValueError, match="metadata.*replay"):
        p.compare_causal_pair(tampered.before, tampered.after)
    with pytest.raises(ValueError, match="metadata.*replay"):
        save_causal_pair(tmp_path / "pair", tampered, max_bytes=1_000_000)
    assert not (tmp_path / "pair").exists()


def test_recipe_json_whitespace_is_not_a_semantic_change() -> None:
    original = pair()
    before = replace(
        original.before, recipe_json=json.dumps(json.loads(original.before.recipe_json), indent=2)
    )
    after = replace(
        original.after, recipe_json=json.dumps(json.loads(original.after.recipe_json), indent=2)
    )
    assert p.compare_causal_pair(before, after) == json.loads(original.card_json)


@pytest.mark.parametrize("side", ["before", "after"])
def test_native_target_origin_is_bound_to_replay(side: str) -> None:
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    base = p.NativeCausalRecipe(
        task,
        (
            p.NativeCausalNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
                ),
                p.VariableDomain(),
            ),
            p.NativeCausalNode(
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))), p.VariableDomain()
            ),
        ),
        ((0, 0),),
        (1, 0),
    )
    original = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=61, task_id=11
    )
    tampered = replace(
        original, **{side: replace(getattr(original, side), target_origin="adapter-v1")}
    )
    with pytest.raises(ValueError, match="metadata.*replay"):
        p.compare_causal_pair(tampered.before, tampered.after)


def test_descriptive_environment_is_retained_without_becoming_replay_authority() -> None:
    original = pair()
    recipe = json.loads(original.before.recipe_json)
    recipe["execution"]["cpu_threads"] += 1
    before = replace(original.before, recipe_json=json.dumps(recipe))
    # Exact array replay checks values; recorded historical thread counts are not a current setting.
    assert p.compare_causal_pair(before, original.after) == json.loads(original.card_json)
