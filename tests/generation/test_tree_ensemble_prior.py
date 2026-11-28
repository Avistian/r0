"""Random ensemble size/weights keep frozen per-task construction provenance."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.generation.lib.tree_ensemble_construction import construct_tree_ensemble


def controls(prior: g.TreeEnsemblePrior) -> dict[str, Any]:
    document = json.loads(Path("configs/generation/sampled-tree-ensemble-cpu.json").read_text())
    value = json.loads(document["construction"]["controls_json"])
    value["weights"] = []
    value["prior"] = asdict(prior)
    return value


@pytest.mark.parametrize("law", ["equal", "dirichlet", "lognormal", "signed-normal"])
def test_sampled_weight_laws_match_explicit_member_scores(law: str) -> None:
    prior = g.TreeEnsemblePrior(min_members=3, max_members=3, weight_law=law)
    config = controls(prior)
    if law == "signed-normal":
        config["reduction"] = "sum"
    construction = g.GraphConstruction(json.dumps(config), 42)
    recipe, receipt = construct_tree_ensemble(
        construction, limits=p.CausalLimits(), seed=29, task_id=5
    )
    again, same = construct_tree_ensemble(construction, limits=p.CausalLimits(), seed=29, task_id=5)
    assert asdict(again) == asdict(recipe)
    assert same == receipt
    assert receipt["version"] == "r0-tree-ensemble-construction-v2"
    assert receipt["prior"] == json.loads(json.dumps(asdict(prior)))
    weights = receipt["weights"]
    assert len(weights) == 3
    if law == "dirichlet":
        assert sum(weights) == pytest.approx(1)
    if law != "signed-normal":
        assert min(weights) > 0
    actual = p.generate_task(recipe, seed=29, task_id=5)
    members = [
        p.generate_task(p.sampled_tree_extension_from_dict(member).recipe, seed=29, task_id=5)
        for member in receipt["members"]
    ]
    for member in members:
        torch.testing.assert_close(member.inputs, actual.inputs, rtol=0, atol=0)
    denominator = sum(weights) if config["reduction"] == "mean" else 1
    expected = sum(
        weight / denominator * task.targets for weight, task in zip(weights, members, strict=True)
    )
    torch.testing.assert_close(actual.targets, expected, rtol=1e-12, atol=1e-12)


def test_count_and_template_support_and_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = g.TreeEnsemblePrior(
        min_members=1, max_members=3, count_probabilities=(0, 0, 1), member_probabilities=(0, 1, 0)
    )
    construction = g.GraphConstruction(json.dumps(controls(prior)), 42)
    _, receipt = construct_tree_ensemble(construction, limits=p.CausalLimits(), seed=29, task_id=0)
    assert receipt["selected_templates"] == [1, 1, 1]
    assert len({r["initial"]["digest"] for r in receipt["members"]}) == 3
    from r0.modules.generation.lib import tree_ensemble_construction as module

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("random draw happened before admission")

    monkeypatch.setattr(module, "sample_ensemble_prior", unexpected)
    with pytest.raises(ValueError, match="node count"):
        construct_tree_ensemble(
            construction, limits=replace(p.CausalLimits(), nodes=40), seed=29, task_id=0
        )
    bad = controls(replace(prior, replacement=False))
    with pytest.raises(ValueError, match="positive-probability"):
        construct_tree_ensemble(
            g.GraphConstruction(json.dumps(bad), 42), limits=p.CausalLimits(), seed=29, task_id=0
        )
    with pytest.raises(ValueError, match="sum to one"):
        g.TreeEnsemblePrior(count_probabilities=(0.1, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="sum reduction"):
        g.TreeEnsemblePrior(weight_law="signed-normal").validate_pool(3, "mean")


def test_guarded_prior_archive_resume_and_realized_count(tmp_path: Path) -> None:
    prior = g.TreeEnsemblePrior(min_members=1, max_members=3, weight_law="dirichlet")
    request = g.request_from_dict(
        {
            "construction": asdict(g.GraphConstruction(json.dumps(controls(prior)), 42)),
            "seed": 29,
            "count": 2,
            "max_seconds": 60,
            "max_bytes": 16000000,
        }
    )
    result = g.run_guarded(request, tmp_path)
    assert result.returncode == 0, result.stderr
    for archive in tmp_path.rglob("data-*.zip"):
        for task in g.load_group(archive):
            receipt = json.loads(task.recipe_json)["construction"]
            assert 1 <= len(receipt["members"]) <= 3
            assert len(receipt["weights"]) == len(receipt["members"])
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
