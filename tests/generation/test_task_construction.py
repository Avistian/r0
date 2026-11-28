"""Per-task mechanisms preserve fixed-mode behavior and archive recovery."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.generation.lib.requests import generate_requested, record_matches
from r0.modules.generation.lib.task_construction import resolve_task_construction


def controls(topology: str) -> dict[str, Any]:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    if topology == "network":
        return {
            "topology": topology,
            "task": asdict(task),
            "network_prior": asdict(p.NetworkSamplingRequest(2, (3, 1), families=("dense",))),
        }
    if topology == "network-hyperprior":
        prior = asdict(
            p.network_source_hyperprior(
                p.NetworkSamplingRequest(2, (1,), families=("dense",)), source="p1-paper"
            )
        )
        prior["depth"] = asdict(p.NetworkScalarPrior("fixed", 2.0, 2.0))
        prior["width"] = asdict(p.NetworkScalarPrior("integer-uniform", 3.0, 6.0))
        return {"topology": topology, "task": asdict(task), "root": None, "hyperprior": prior}
    common = {
        "topology": topology,
        "task": asdict(task),
        "calibration_rows": 16,
        "root_families": ["gaussian"],
        "function_prior": asdict(
            p.JointFunctionPrior(functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
        ),
    }
    if topology in ("explicit", "cauchy"):
        return {
            **common,
            "widths": [2, 2, 1],
            "observed": [[0, 0], [0, 1]],
            "target": [[2, 0]],
            **({"parents": [[], [0], [0, 1]]} if topology == "explicit" else {"offset": 0.0}),
        }
    geometry = {
        **common,
        "subsample": True,
        "width_prior": asdict(p.JointWidthPrior()),
        "noise_scale": 0.0,
    }
    if topology == "geometry":
        return {**geometry, "nodes": 3, "offset": 0.0}
    prior = (
        p.AttachmentGraphPrior(nodes=3)
        if topology == "attachment-geometry"
        else p.MotifGraphPrior(depth=1)
    )
    return {**geometry, "topology_prior": asdict(prior)}


def request(topology: str, *, fresh: bool = True) -> g.RunRequest:
    source = g.GraphConstruction(json.dumps(controls(topology)), 42)
    return g.request_from_dict(
        {
            "construction": asdict(g.per_task_construction(source) if fresh else source),
            "seed": 29,
            "count": 1,
            "max_seconds": 60,
            "max_bytes": 16000000,
        }
    )


@pytest.mark.parametrize(
    "topology",
    [
        "explicit",
        "cauchy",
        "geometry",
        "attachment-geometry",
        "motif-geometry",
        "network",
        "network-hyperprior",
    ],
)
def test_realized_functions_change_and_frozen_recipe_replays(topology: str) -> None:
    fixed = request(topology, fresh=False)
    old = [generate_requested(fixed, i) for i in (0, 1)]
    assert json.loads(old[0].recipe_json)["recipe"] == json.loads(old[1].recipe_json)["recipe"]
    sampled = request(topology)
    tasks = [generate_requested(sampled, i) for i in (1, 0, 1)]
    records = [json.loads(t.recipe_json) for t in tasks]
    assert records[0] == records[2]
    # Recipe fields contain frozen mechanisms, not the new sampling receipt.
    assert records[0]["recipe"] != records[1]["recipe"]
    # Common episode randomness makes a changed response evidence of an actual
    # mechanism change, rather than just a different ID in the receipt.
    replay = [
        p.generate_task(p.recipe_from_dict(r["recipe"]), seed=17, task_id=0) for r in records[:2]
    ]
    assert not torch.equal(replay[0].targets, replay[1].targets)
    for task, record in zip(tasks, records, strict=True):
        rebuilt = p.generate_task(
            p.recipe_from_dict(record["recipe"]), seed=29, task_id=record["task_id"]
        )
        torch.testing.assert_close(task.inputs, rebuilt.inputs, rtol=0, atol=0)
        torch.testing.assert_close(task.targets, rebuilt.targets, rtol=0, atol=0)
        assert record_matches(record, sampled, record["task_id"])
        p.explain_generation(task, [0])
    assert g.CoveragePlan((g.CoverageStratum("fresh", sampled),), total=100_000_000)
    with pytest.raises(ValueError, match="independent-source"):
        g.CoveragePlan((g.CoverageStratum("fixed", fixed),), total=2)


def test_receipt_binding_and_seed_domain() -> None:
    sampled = request("network")
    assert sampled.construction is not None
    ids = [*range(64), 2**54 + 3, 2**63 - 1]
    seeds = [
        resolve_task_construction(sampled.construction, episode_seed=29, task_id=i)[0].seed
        for i in ids
    ]
    assert len(set(seeds)) == len(ids)
    for invalid in (True, -1, 2**63):
        with pytest.raises(ValueError, match="bounded integers"):
            resolve_task_construction(sampled.construction, episode_seed=29, task_id=invalid)
    raw = json.loads(generate_requested(sampled, 0).recipe_json)
    for key in raw["construction_sampling"]:
        changed = json.loads(json.dumps(raw))
        changed["construction_sampling"][key] = "tampered"
        assert not record_matches(changed, sampled, 0)
    effective, _ = resolve_task_construction(sampled.construction, episode_seed=29, task_id=0)
    assert not record_matches(raw, replace(sampled, construction=effective), 0)
    del raw["construction_sampling"]
    assert not record_matches(raw, sampled, 0)
    assert record_matches(raw, replace(sampled, construction=effective), 0)
    for topology in ("network-authored", "native", "unknown", "per-task"):
        with pytest.raises(ValueError, match="not qualified"):
            g.per_task_construction(g.GraphConstruction(json.dumps({"topology": topology}), 42))


def test_guarded_observed_campaign_resumes(tmp_path: Path) -> None:
    sampled = replace(
        request("network"),
        observation=g.ObservationRequest(
            p.MissingnessPlan("mcar", (0,), probability=0.25), seed=42
        ),
    )
    plan = g.CoveragePlan((g.CoverageStratum("networks", sampled),), total=2, shard_size=2)
    result = g.run_coverage_shard(plan, 0, tmp_path)
    assert result.returncode == 0, result.stderr
    archives = sorted(tmp_path.rglob("data-*.zip"))
    before = [a.read_bytes() for a in archives]
    tasks = [t for archive in archives for t in g.load_group(archive)]
    assert len(tasks) == 2
    for task in tasks:
        raw = json.loads(task.recipe_json)
        assert record_matches(raw, sampled, raw["task_id"])
    resumed = g.run_coverage_shard(plan, 0, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
    assert before == [a.read_bytes() for a in archives]
