"""Public sampled-law and actual saved-session evidence; output path is a CLI parameter."""

import argparse
import hashlib
import inspect
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import (
    CausalLimits,
    ImportanceWeights,
    LinearRecipe,
    MatrixCausalNode,
    PolynomialWeightVector,
    UniformWeightVector,
    recipe_from_dict,
    sample_recipe,
)


def dump(p, v):
    p.write_text(json.dumps(v, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    counts = {"polynomial": 0, "uniform": 0, "low": 0, "high": 0, "interior": 0, "rows": 0}
    gaussian = []
    uniforms = []
    z = []
    logscale = []
    params = []
    perms = set()
    widths = set()
    checks = 0
    for seed in range(128):
        for family in ("forward", "layered", "redirection"):
            r = sample_recipe(
                family,
                task=LinearRecipe(rows=16, context_rows=8),
                seed=seed,
                task_id=10,
                profile="causal-importance-v1",
            )
            assert r.construction == "activation-importance-v1"
            checks += 1
            dump(out / f"sampled-{family}-{seed}.json", asdict(r))
            assert asdict(recipe_from_dict(json.loads(json.dumps(asdict(r))))) == asdict(r)
            checks += 1
            for node in r.nodes:
                widths.add(len(node.bias))
                if not node.parents:
                    continue
                assert isinstance(node, MatrixCausalNode)
                assert type(node.weight_construction) is ImportanceWeights
                checks += 1
                wc = node.weight_construction
                kinds = {type(v) for v in wc.importance}
                assert len(kinds) == 1
                checks += 1
                counts["polynomial" if kinds == {PolynomialWeightVector} else "uniform"] += 1
                assert wc.epsilon == 1e-6
                checks += 1
                for row, v in zip(wc.gaussian, wc.importance, strict=True):
                    gaussian.extend(row)
                    counts["rows"] += 1
                    if isinstance(v, UniformWeightVector):
                        uniforms.extend(v.draws)
                        continue
                    low = 0.1 / math.log(1 + len(v.draws))
                    assert low <= v.decay <= 6
                    assert 1e-4 <= v.log_noise_scale <= 10
                    checks += 1
                    counts["low" if v.decay == low else "high" if v.decay == 6 else "interior"] += 1
                    logscale.append(math.log(v.log_noise_scale / 1e-4) / math.log(1e5))
                    z.extend(v.draws)
                    perms.add(v.permutation)
                    params.append((v.decay, v.log_noise_scale, v.draws, v.permutation))
    total = counts["polynomial"] + counts["uniform"]
    poly = sum(counts[k] for k in ("low", "high", "interior"))

    def stat(xs):
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "variance": statistics.pvariance(xs),
            "min": min(xs),
            "max": max(xs),
        }

    stats = dict(
        counts=counts,
        widths=sorted(widths),
        distinct_permutations=len(perms),
        gaussian=stat(gaussian),
        uniform=stat(uniforms),
        z=stat(z),
        logscale=stat(logscale),
    )
    # Broad diagnostic intervals; exact sampler-law source inspection remains separate.
    assert 0.43 < counts["polynomial"] / total < 0.57
    assert 0.055 < counts["low"] / poly < 0.115
    assert 0.055 < counts["high"] / poly < 0.115
    assert abs(stat(gaussian)["mean"]) < 0.1
    assert 0.8 < stat(gaussian)["variance"] < 1.2
    assert abs(stat(z)["mean"]) < 0.1
    assert 0.8 < stat(z)["variance"] < 1.2
    assert 0.46 < stat(uniforms)["mean"] < 0.54
    assert 0.46 < stat(logscale)["mean"] < 0.54
    assert widths == {1, 2, 4}
    checks += 6
    dump(out / "sampled-law-statistics.json", stats)
    workflows = []
    for family, seed in (("forward", 28), ("layered", 28), ("redirection", 28), ("layered", 0)):
        directory = out / f"explorer-{family}-{seed}"
        directory.mkdir()
        session = ExplorerSession(directory / "saved")
        result = session.request(
            "generate",
            {
                "record": {"recipe": {"rows": 16, "context_rows": 8}, "seed": seed, "task_id": 10},
                "sampling": {"family": family, "profile": "causal-importance-v1"},
            },
        )
        dump(directory / "generated.json", result)
        graph = session.request("graph", {"indices": [0, 15]})["graph"]
        dump(directory / "graph.json", graph)
        assert graph["target_max_error"] < 1e-10
        checks += 1
        constructions = [
            n["parameters"]["weight_construction"]
            for n in graph["nodes"]
            if "weight_construction" in n.get("parameters", {})
        ]
        expected_nodes = json.loads(result["recipe_json"])["recipe"]["nodes"]
        assert len(constructions) == sum(bool(n["parents"]) for n in expected_nodes)
        assert all(v["version"] == "importance-weights-v1" for v in constructions)
        checks += 1
        reopened = ExplorerSession(directory / "saved").request(
            "open", {"archive": result["archive"], "task_id": 10}
        )
        dump(directory / "reopened.json", reopened)
        assert reopened["view"] == result["view"]
        checks += 1
        replay_session = ExplorerSession(directory / "replay")
        replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
        dump(directory / "replay.json", replay)
        assert replay["recipe_json"] == result["recipe_json"]
        checks += 1
        for index in (0, 15):
            row = session.request("row", {"index": index})
            dump(directory / f"row-{index}.json", row)
            assert replay_session.request("row", {"index": index}) == row
            checks += 1
        workflows.append(
            {
                "family": family,
                "seed": seed,
                "status": "PASS",
                "saved_archive": result["archive"],
                "trace_target_max_error": graph["target_max_error"],
            }
        )
    checks += admission(out)
    receipt = dict(
        status="PASS",
        checks=checks,
        recipes=384,
        workflows=workflows,
        python=sys.executable,
        public_import=inspect.getfile(ImportanceWeights),
    )
    dump(out / "workflow-summary.json", receipt)
    print(json.dumps(receipt))


def admission(out):
    events = []
    original_rng = np.random.default_rng
    original_admit = CausalLimits.admit_dimensions

    def digest(a):
        return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()

    class LoggedRng:
        def __init__(self, rng):
            self.rng = rng

        def __getattr__(self, name):
            fn = getattr(self.rng, name)
            if name != "normal":
                return fn

            def call(*args, **kwargs):
                value = fn(*args, **kwargs)
                events.append(
                    {"kind": "normal", "shape": list(np.shape(value)), "sha256": digest(value)}
                )
                return value

            return call

    def rng(*args, **kwargs):
        return LoggedRng(original_rng(*args, **kwargs))

    def admit(self, *args, **kwargs):
        result = original_admit(self, *args, **kwargs)
        events.append({"kind": "admission", "result": result})
        return result

    with (
        patch.object(np.random, "default_rng", rng),
        patch.object(CausalLimits, "admit_dimensions", admit),
    ):
        recipe = sample_recipe(
            "forward",
            task=LinearRecipe(rows=16, context_rows=8),
            seed=28,
            task_id=10,
            profile="causal-importance-v1",
        )
    expected = recipe.limits.admit(
        recipe.nodes, rows=16, calibration_rows=0, features=recipe.task.features, outputs=1
    )
    admissions = [
        i for i, e in enumerate(events) if e["kind"] == "admission" and e["result"] == expected
    ]
    assert admissions
    allocation = []
    for node in recipe.nodes:
        if not node.parents:
            continue
        hashed = digest(node.weight_construction.gaussian)
        matched = [
            i for i, e in enumerate(events) if e["kind"] == "normal" and e["sha256"] == hashed
        ]
        assert len(matched) == 1
        assert admissions[0] < matched[0]
        allocation.extend(matched)
    assert allocation
    dump(
        out / "sampling-admission.json",
        {
            "events": events,
            "expected": expected,
            "matching_admissions": admissions,
            "gaussian_allocation_events": allocation,
        },
    )
    return len(allocation) + 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    run(p.parse_args().output_dir)
