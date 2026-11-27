"""Matching archives retain one raw record per source coordinate, including wide roots."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("width", [3, 6, 256])
def test_source_root_matching_roundtrip_and_original_row_trace(width: int):
    coordinates = (
        p.SourceRootCoordinate("gaussian"),
        p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.3, 0.5)),
        p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),
    )
    law = p.SourceRootLaw(tuple(coordinates[i % 3] for i in range(width)))
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=width, signal_features=width, active_features=width
    )
    network = p.FrozenNetworkFunction(
        width, (p.NetworkBlock((0,), (p.NetworkAffine(((1 / width,) * width,), (0.0,)),)),)
    )
    base = p.NetworkRecipe(controls, network, p.SourceRootBase(law))
    after = tuple(range(1, width))
    match = p.fit_network_variance_match(
        base,
        before_inputs=tuple(range(width)),
        after_inputs=after,
        rows=31,
        rng=np.random.default_rng(87),
    )
    assert len(match.raw) == width
    view = p.MatchedNetworkInputView(
        base, replace(controls, active_features=width - 1), after, matching=match
    )
    assert p.recipe_from_dict(asdict(view)) == view
    task = p.generate_task(view, seed=7, task_id=3)
    trace = p.explain_generation(task, [15, 0, 15])
    assert trace["signal_max_error"] == 0
    tampered = asdict(view)
    tampered["matching"]["raw"] = (*tampered["matching"]["raw"], tampered["matching"]["raw"][0])
    with pytest.raises(ValueError, match="oversized"):
        p.recipe_from_dict(tampered)
