"""Whole task admission precedes any recursive construction RNG restoration."""

from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def recipes():
    plan = p.RecursiveForestPlan(
        input_width=1, estimators=3, min_depth=1, max_depth=2, intervals=((-2.0, 2.0),)
    )
    sample = p.sample_recursive_forest(plan, rng=np.random.default_rng(19))
    function = p.FrozenRecursiveForestFunction(sample)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    child = p.FunctionalCausalNode((0,), function)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    limits = p.CausalLimits(working_bytes=32 * 1024**2)
    functional = p.FunctionalCausalRecipe(task, (root, child), ((0, 0),), ((1, 0),), limits)
    native = p.NativeCausalRecipe(
        task, (p.NativeCausalNode(root), p.NativeCausalNode(child)), ((0, 0),), (1, 0), limits
    )
    vector = p.NativeVectorRecipe(
        task,
        (
            p.NativeVectorNode(root, (p.VariableDomain(),)),
            p.NativeVectorNode(child, (p.VariableDomain(),)),
        ),
        ((0, 0),),
        (1, 0),
        limits,
    )
    converted = p.ConvertedCausalRecipe(
        task,
        (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
        (p.ConvertedReadout(0, channel=0),),
        p.ConvertedReadout(1, channel=0),
        limits,
    )
    parent = replace(
        functional,
        nodes=(root, replace(child, function=p.FrozenParentFunction((function,), "mean"))),
    )
    return sample, (functional, native, vector, converted, parent)


@pytest.mark.parametrize("index", range(5))
def test_rows_reject_before_any_restore(index: int):
    sample, fixtures = recipes()
    raw = asdict(fixtures[index])
    raw["task"]["rows"] = 20000
    with (
        patch.object(
            type(sample.initial),
            "restore",
            side_effect=AssertionError("restore before whole task admission"),
        ),
        pytest.raises(ValueError, match="memory|workspace"),
    ):
        p.recipe_from_dict(raw)


@pytest.mark.parametrize("index", range(5))
def test_valid_whole_task_reopens_without_custody_changes(index: int):
    _, fixtures = recipes()
    original = fixtures[index]
    assert p.recipe_from_dict(asdict(original)) == original
    task = p.generate_task(original, seed=19, task_id=2**54 + 3)
    replay = p.generate_task(p.recipe_from_dict(asdict(original)), seed=19, task_id=2**54 + 3)
    assert replay.recipe_json == task.recipe_json
    assert replay.mechanism_json == task.mechanism_json


@pytest.mark.parametrize("root_kind", ["sampled-source", "prototype"])
def test_root_budgets_and_width_precede_forest_restore(root_kind: str):
    sample, fixtures = recipes()
    original = fixtures[0]
    assert isinstance(original, p.FunctionalCausalRecipe)
    if root_kind == "sampled-source":
        profile = p.sample_source_root_profile(
            p.SourceRootProfile(1, selector="forced-r0", families=("gaussian",)),
            model_rng=np.random.default_rng(3),
            selection_rng=np.random.default_rng(5),
        )
        root = p.sample_source_root_base(profile, modes=("identity",))
    else:
        root = p.PrototypeRootBase(p.FrozenPrototypeRoot(((-1.0,), (2.0,))))
    original = replace(
        original,
        nodes=(
            replace(original.nodes[0], root=root, version="functional-causal-node-v2"),
            original.nodes[1],
        ),
    )
    assert p.recipe_from_dict(asdict(original)) == original
    raw = asdict(original)
    raw["task"]["rows"] = 20000
    with (
        patch.object(
            type(sample.initial),
            "restore",
            side_effect=AssertionError("root restored before whole task admission"),
        ),
        pytest.raises(ValueError, match="workspace|memory"),
    ):
        p.recipe_from_dict(raw)


@pytest.mark.parametrize("field", ["root", "parents", "observed"])
def test_known_structural_geometry_rejects_before_restore(field: str):
    sample, fixtures = recipes()
    raw = asdict(fixtures[0])
    if field == "root":
        raw["nodes"][0]["root"]["width"] = 2
    elif field == "parents":
        raw["nodes"][1]["parents"] = (1,)
    else:
        raw["observed"] = ()
    with (
        patch.object(
            type(sample.initial),
            "restore",
            side_effect=AssertionError("restore before geometry admission"),
        ),
        pytest.raises(ValueError, match="geometry|width"),
    ):
        p.recipe_from_dict(raw)
