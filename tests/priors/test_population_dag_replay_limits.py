"""Complete population pairs obey the same declared replay budget as construction."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


def test_two_output_population_pair_roundtrips_original_default_limits(
    monkeypatch: pytest.MonkeyPatch,
):
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((2.0, -3.0),))),
        ),
        ((0, 0), (0, 1)),
        ((2, 0),),
    )
    forest = p.RecursiveForestPlan(
        input_width=2,
        estimators=2,
        min_depth=2,
        max_depth=3,
        leaf_mean=(0.5, -1.0),
        leaf_scale=(0.4, 1.2),
        source="r0-extension-v1",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    population = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, forest),)),
        seed=43,
        task_id=2**55 + 31,
        rng=np.random.default_rng(47),
    )
    change = p.CausalInterventionRecipe(
        base,
        (p.HardIntervention(0, (0, 1), (0.25, -0.75)),),
        mode="matched",
        population=7,
    )
    changed = p.intervene_recursive_population_graph(population, change)
    restored = p.recipe_from_dict(asdict(changed))
    assert asdict(restored) == asdict(changed)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("over-budget pair restored a generator")

    monkeypatch.setattr(type(population.bindings[0].sample.initial), "restore", forbidden)
    for mb in (1, 64, 128, 224):
        with pytest.raises(ValueError, match="limit|workspace|metadata|reservation"):
            p.recipe_from_dict(asdict(changed), limits=p.CausalLimits(working_bytes=mb * 1024**2))
