"""Archive-bound observation curve delivery and guarded failure preservation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def construct(session: ExplorerSession):
    controls = session.request("construction_examples", {})["examples"][
        "Observations: authored affine equations / pinned selection"
    ]
    controls["selection_seed"] = "9007199254741006"
    controls["observations"]["feature_order"] = "permuted"
    return session.request(
        "construct",
        {"controls": controls, "construction_seed": "7", "seed": "37", "task_id": str(2**54 + 17)},
    )


@pytest.mark.parametrize("reference", ["zero", "original-row"])
def test_observation_curve_retains_full_identity_and_alias_law(
    tmp_path: Path, reference: str
) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session)
    recipe = p.recipe_from_dict(json.loads(baseline["recipe_json"])["recipe"])
    assert isinstance(recipe, p.NetworkObservationRecipe)
    selected = recipe.observations.features.index(recipe.observations.targets[0])
    curve = session.request(
        "curve", {"feature": selected, "points": 9, "anchor_row": 31, "reference": reference}
    )["curve"]
    assert curve["original_task_id"] == str(2**54 + 17)
    assert curve["recipe_json"] == baseline["recipe_json"]
    assert curve["observation_selection"]["target_aliases"]
    assert curve["clamp_location"] == "state_and_readout"
    # The target alias must equal the clamped original observed channel exactly.
    np.testing.assert_array_equal(np.array(curve["scores"])[:, 0], curve["values"])
    assert session.request("view", {}) == baseline


def test_observation_curve_decode_is_guarded_and_failures_preserve_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session)
    recipe = p.recipe_from_dict(json.loads(baseline["recipe_json"])["recipe"])
    assert isinstance(recipe, p.NetworkObservationRecipe)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("observation source replay escaped the worker")

    monkeypatch.setattr(type(recipe.observations.initial), "restore", forbidden)
    session.request("curve", {"feature": 0, "points": 3})
    for extra in (
        {"max_seconds": 0.001},
        {"points": 513},
        {"reference": "conditional-mean"},
        {"max_bytes": 1},
        {"bounds": [2.0, 1.0]},
        {"anchor_row": 32},
    ):
        with pytest.raises(ValueError, match="timed out|failed"):
            session.request("curve", {"feature": 0, "points": 3, **extra})
        assert session.request("view", {}) == baseline


def test_curve_validates_full_saved_envelope_metadata_not_only_runtime(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    generated = construct(session)
    task = next(g.load_group(tmp_path / generated["archive"]))
    altered = replace(task, task_type="binary")
    path = tmp_path / "altered.zip"
    g.save_group(path, [(2**54 + 17, altered)], max_bytes=256 * 1024**2)
    opened = session.request("open", {"archive": path.name, "task_id": str(2**54 + 17)})
    with pytest.raises(ValueError, match="semantic metadata"):
        session.request("curve", {"feature": 0, "points": 3})
    assert session.request("view", {}) == opened


@pytest.mark.parametrize(
    "label",
    [
        "P1 paper + explicit r0 laws / individual-sparse",
        "P1 pinned 778f0d0 / block-sparse",
        "internal noise + importance / pre-activation (r0)",
        "P1 scalar weight prior / explicit fixed equal widths",
        "p1-code-mixed coordinate root / identity",
        "independent prototype root / r0 count and weight laws",
    ],
)
def test_frozen_source_compositions_have_exact_anchor_reference(tmp_path: Path, label: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"]["Observations: " + label]
    recipe = g.GraphConstruction(json.dumps(controls), 7).build(p.CausalLimits())
    assert isinstance(recipe, p.NetworkObservationRecipe)
    task = p.generate_task(recipe, seed=37, task_id=2**54 + 17)
    import torch

    reference = p.evaluate_observation_reference(
        recipe, task.inputs[5:6], seed=37, task_id=2**54 + 17, row=5
    )
    torch.testing.assert_close(reference, task.scores[5:6], rtol=0, atol=1e-12)
