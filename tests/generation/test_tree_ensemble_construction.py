"""Per-task mixed tree ensembles retain independent member draws and resume identity."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.generation.lib.requests import record_matches


def configuration() -> dict[str, object]:
    task = p.LinearRecipe(
        rows=32, context_rows=16, features=2, signal_features=2, active_features=2
    )
    root = p.SourceRootBase(
        p.SourceRootLaw(
            (
                p.SourceRootCoordinate("gaussian"),
                p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.3, 0.5)),
            )
        )
    )
    members = [
        asdict(
            p.TreeExtensionRequest(task, root=root, families=(family,), min_depth=1, max_depth=2)
        )
        for family in ("hard", "soft", "categorical-pruned")
    ]
    return {
        "topology": "tree-ensemble",
        "members": members,
        "weights": [1.0, 2.0, 3.0],
        "reduction": "mean",
    }


def test_guarded_ensemble_receipts_outputs_and_resume(tmp_path: Path) -> None:
    construction = GraphConstruction(json.dumps(configuration()), 42)
    request = g.request_from_dict(
        {
            "construction": asdict(construction),
            "seed": 29,
            "count": 2,
            "max_seconds": 60,
            "max_bytes": 16000000,
        }
    )
    result = g.run_guarded(request, tmp_path, workers=2)
    assert result.returncode == 0, result.stderr
    assert result.summary is not None
    assert result.summary["completed"] == 2
    count = 0
    for path in tmp_path.rglob("data-*.zip"):
        for task in g.load_group(path):
            count += 1
            raw = json.loads(task.recipe_json)
            identity = raw["task_id"]
            receipts = raw["construction"]["members"]
            assert len({r["initial"]["digest"] for r in receipts}) == 3
            members = [p.sampled_tree_extension_from_dict(r) for r in receipts]
            individual = [p.generate_task(m.recipe, seed=29, task_id=identity) for m in members]
            for member in individual:
                torch.testing.assert_close(member.inputs, task.inputs, rtol=0, atol=0)
            expected = sum(
                w * m.targets for w, m in zip((1 / 6, 2 / 6, 3 / 6), individual, strict=True)
            )
            torch.testing.assert_close(task.targets, expected, rtol=1e-13, atol=1e-13)
            assert record_matches(raw, request, identity)
            altered = deepcopy(raw)
            altered["construction"]["weights"][0] = 9.0
            assert not record_matches(altered, request, identity)
            altered = deepcopy(raw)
            altered["construction"]["members"].reverse()
            assert not record_matches(altered, request, identity)
    assert count == 2
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0


@pytest.mark.parametrize("invalid", ["weights", "task", "nodes"])
def test_ensemble_preflight_precedes_sampling(
    invalid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r0.modules.generation.lib import tree_ensemble_construction as module

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("sampled before validating ensemble controls")

    monkeypatch.setattr(module, "sample_tree_extension", unexpected)
    controls = json.loads(json.dumps(configuration()))
    limits = p.CausalLimits()
    if invalid == "weights":
        controls["weights"] = [1.0, -1.0, 0.0]
    elif invalid == "task":
        controls["members"][1]["task"]["rows"] = 33
    else:
        limits = replace(limits, nodes=40)
    with pytest.raises(ValueError, match="weight|share task|node count"):
        module.construct_tree_ensemble(
            GraphConstruction(json.dumps(controls), 42), limits=limits, seed=29, task_id=0
        )
