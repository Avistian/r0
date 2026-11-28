"""Estimated matching stays inside the killable pair worker and persisted replay."""

import json
from dataclasses import asdict
from pathlib import Path

from r0.modules import generation as g
from r0.modules import priors as p


def test_guarded_matching_pair_and_failed_recalibration(tmp_path: Path):
    recipe = p.NetworkRecipe(
        p.LinearRecipe(rows=21, context_rows=9, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
        ),
    )
    request = g.RunRequest(
        recipe, seed=7, start=2**54 + 3, count=1, max_seconds=30, max_bytes=4_000_000
    )
    before_run = g.run_guarded(request, tmp_path / "before")
    assert before_run.returncode == 0, before_run.stderr
    before_path = next((tmp_path / "before").glob("groups/**/*.zip"))
    original_bytes = before_path.read_bytes()
    data = asdict(request)
    data["pairing"] = {
        "before_archive": str(before_path),
        "intervention": "sparsity",
        "value": 1,
        "allow_estimated_variance": True,
    }
    after_run = g.run_guarded_document(data, tmp_path / "after")
    assert after_run.returncode == 0, after_run.stderr
    assert after_run.summary is not None
    assert "independent_network_baseline_moments_estimate" in json.dumps(after_run.summary)
    after_path = next((tmp_path / "after").glob("groups/**/*.zip"))
    after = next(g.load_group(after_path))
    before = next(g.load_group(before_path))
    assert p.compare_pair(before, after, intervention="sparsity", allow_estimated_variance=True)
    data["recipe"] = json.loads(after.recipe_json)["recipe"]
    data["pairing"]["value"] = 2
    data["pairing"]["before_archive"] = str(after_path)
    failed = g.run_guarded_document(data, tmp_path / "rejected")
    assert failed.returncode != 0
    assert "already matched" in failed.stderr
    assert before_path.read_bytes() == original_bytes
