"""Sampled causal tasks travel through the shared saved explorer workflow."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from r0.modules import priors
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["forward", "layered", "redirection"])
@pytest.mark.parametrize(
    ("profile", "construction"),
    [
        ("causal-affine-v1", "gaussian-affine-v1"),
        ("causal-activated-v1", "gaussian-activated-v1"),
        ("causal-preferences-v1", "gaussian-activation-prior-v1"),
        ("causal-exogenous-v1", "activation-exogenous-v1"),
        ("causal-factorized-v1", "activation-factorized-v1"),
        ("causal-importance-v1", "activation-importance-v1"),
        ("causal-matrices-v1", "activation-matrices-v1"),
        ("causal-gaussian-matrices-v1", "activation-gaussian-matrices-v1"),
        ("causal-shared-matrices-v1", "activation-shared-matrices-v1"),
        ("causal-refreshed-matrices-v1", "activation-shared-matrices-v1"),
        ("causal-local-matrices-v1", "activation-shared-matrices-v1"),
        ("causal-global-matrices-v1", "activation-shared-matrices-v1"),
        ("causal-source-standard-v1", "source-activations-v1"),
        ("causal-source-standard-refreshed-v1", "source-activations-v1"),
        ("causal-source-standard-local-v1", "source-activations-v1"),
        ("causal-source-standard-global-v1", "source-activations-v1"),
        ("causal-source-plain-v1", "source-activations-v1"),
        ("causal-source-plain-refreshed-v1", "source-activations-v1"),
        ("causal-source-plain-local-v1", "source-activations-v1"),
        ("causal-source-plain-global-v1", "source-activations-v1"),
    ],
)
def test_sampled_causal_session_saves_reopens_and_replays_its_graph(
    tmp_path: Path, family: str, profile: str, construction: str
) -> None:
    session = ExplorerSession(tmp_path / "original")
    request = {
        "record": {
            "recipe": {"rows": 16, "context_rows": 8},
            "seed": 28,
            "task_id": 10,
        },
        "sampling": {"family": family, "profile": profile},
    }
    if profile == "causal-source-plain-refreshed-v1" and family in ("forward", "layered"):
        # The original unfiltered profile pass exposed these two exp-overflow
        # draws. Keep their exact identities and full recipes; do not replace them.
        recipe = priors.sample_recipe(
            family,
            profile=profile,
            task=priors.LinearRecipe(rows=16, context_rows=8),
            seed=28,
            task_id=10,
        )
        (tmp_path / "failed-recipe.json").write_text(json.dumps(asdict(recipe)))
        with pytest.raises(ValueError, match="Generation failed"):
            session.request("generate", request)
        failures = list((tmp_path / "original").rglob("failures-0.jsonl"))
        assert len(failures) == 1
        assert json.loads(failures[0].read_text()) == {
            "task_id": 10,
            "error": "ValueError: nonfinite exp activation output; task was not retried",
        }
        return
    result = session.request("generate", request)
    record = json.loads(result["recipe_json"])
    assert record["recipe"]["topology"]["prior"]["family"] == family
    assert record["recipe"]["construction"] == construction
    trace = session.request("graph", {"indices": [0, 15]})["graph"]
    assert trace["target_max_error"] < 1e-10
    assert any(node["id"] == "causal-0" for node in trace["nodes"])
    reopened = ExplorerSession(tmp_path / "original").request(
        "open", {"archive": result["archive"], "task_id": 10}
    )
    assert reopened["view"] == result["view"]
    replay_session = ExplorerSession(tmp_path / "replay")
    replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
    assert replay["recipe_json"] == result["recipe_json"]
    assert replay_session.request("row", {"index": 15}) == session.request("row", {"index": 15})
