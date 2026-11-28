"""Displayed native emission metadata follows the effective intervened domain."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("weighted", [False, True])
def test_stage_view_reports_effective_emission_and_original_law(tmp_path: Path, weighted: bool):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (authored)"
    ]
    controls["task"].update(rows=16, context_rows=8)
    before = session.request(
        "construct",
        {"controls": controls, "construction_seed": 0, "seed": 7, "task_id": 3, "max_seconds": 30},
    )
    base = p.recipe_from_dict(json.loads(before["recipe_json"])["recipe"])
    assert isinstance(base, p.NativeCausalRecipe)
    node = base.target[0]
    original = base.nodes[node].domain
    domain = (
        p.FrequencyVariableDomain(
            original.kind, original.vocabulary, "softmax", frequencies=(0.1, 0.2, 0.7)
        )
        if weighted
        else replace(original, emission="argmax", temperature=1)
    )
    result = session.request(
        "intervene",
        {
            "interventions_json": json.dumps([asdict(p.StageIntervention(node, domain=domain))]),
            "mode": "matched",
            "population": 1,
            "max_seconds": 30,
        },
    )
    view = result["intervention"]["after"]
    assert view["target_emission"] == domain.emission
    assert view["target_frequency_weights"] == (
        list(domain.frequencies) if isinstance(domain, p.FrequencyVariableDomain) else None
    )
    assert view["original_target_emission"] == original.emission
    assert view["original_target_frequency_weights"] == (
        list(original.frequencies) if isinstance(original, p.FrequencyVariableDomain) else None
    )
    assert result["recipe_json"] == before["recipe_json"]
