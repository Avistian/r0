"""Bounded sampled contracts, independent scalar counting, actual saved backend workflows."""

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import (
    CausalLimits,
    GaussianWeights,
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
    LinearRecipe,
    NormalizedWeights,
    TransformedWeights,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)

PROFILES = ("causal-matrices-v1", "causal-gaussian-matrices-v1")
FAMILIES = ("forward", "layered", "redirection")


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def scalars(value):
    # Count all retained numeric leaves, including assignments/permutations and duplicates.
    # Versions/names/work controls are metadata; no production parameter_count helpers.
    if isinstance(value, dict):
        return sum(
            scalars(v)
            for k, v in value.items()
            if k not in ("exact_terms_limit", "exact_scores_limit", "exact_distance_terms_limit")
        )
    if isinstance(value, (tuple, list)):
        return sum(scalars(v) for v in value)
    return int(type(value) in (int, float))


def run(out):
    out.mkdir(parents=True)
    count = 0
    kinds = Counter()
    laws = Counter()
    children = Counter()
    widths = set()
    outcomes = []
    for profile in PROFILES:
        for family in FAMILIES:
            for task_type, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
                for seed in (0, 1, 28, 45):
                    key = f"{profile}-{family}-{task_type}-{seed}"
                    recipe = None
                    try:
                        recipe = sample_recipe(
                            family,
                            task=LinearRecipe(
                                rows=16, context_rows=8, task_type=task_type, classes=classes
                            ),
                            seed=seed,
                            task_id=10,
                            profile=profile,
                        )
                        dump(out / (key + ".json"), asdict(recipe))
                        assert asdict(
                            recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
                        ) == asdict(recipe)
                        count += 1
                        for node in recipe.nodes:
                            widths.add(len(node.bias))
                            if not node.parents:
                                continue
                            record = node.weight_construction
                            assert type(record) is NormalizedWeights
                            assert record.epsilon == record.jitter_scale == 1e-6
                            assert record.parameter_count == scalars(asdict(record))
                            assert record.materialize() == node.weights
                            count += 4
                            raw = record.source
                            kinds[type(raw).__name__] += 1
                            if profile == PROFILES[1]:
                                assert type(raw) is GaussianWeights
                                count += 1
                            if isinstance(raw, TransformedWeights):
                                assert type(raw.child) is NormalizedWeights
                                assert not isinstance(raw.child.source, TransformedWeights)
                                assert raw.jitter_scale == 1e-3
                                assert raw.child.jitter_scale == raw.child.epsilon == 1e-6
                                children[type(raw.child.source).__name__] += 1
                                raw = raw.child.source
                                count += 4
                            if isinstance(raw, ImportanceFactorizedWeights):
                                assert len(raw.diagonal.draws) == min(raw.shape)
                                count += 1
                            if isinstance(raw, ImportanceWeights):
                                assert len({type(v) for v in raw.importance}) == 1
                                count += 1
                            if isinstance(raw, LaplaceKernelWeights):
                                cloud = raw.point_construction
                                assert cloud is not None
                                assert cloud.shape == (sum(raw.shape), 3)
                                assert 0.1 <= raw.distance_scale <= 10
                                assert cloud.parameter_count == scalars(asdict(cloud))
                                assert len({type(v) for v in cloud.factor_weights}) == 1
                                assert all(
                                    cloud.component_probabilities()[i] > 0
                                    for i in cloud.assignments
                                )
                                assert 1 <= len(cloud.centers) <= 15
                                laws[(cloud.law, cloud.base_law)] += 1
                                count += 7
                        task = generate_task(recipe, seed=seed, task_id=10)
                        replay = generate_task(
                            recipe_from_dict(asdict(recipe)), seed=seed, task_id=10
                        )
                        assert task.episode() == replay.episode()
                        count += 1
                        outcomes.append(dict(key=key, status="PASS"))
                    except ValueError as e:
                        outcomes.append(
                            dict(
                                key=key,
                                status="FAILED",
                                stage="sampling" if recipe is None else "generation",
                                error=str(e),
                            )
                        )
                    dump(out / "outcomes.json", outcomes)
    assert widths == {1, 2, 4}
    assert set(kinds) == {
        "GaussianWeights",
        "ImportanceWeights",
        "ImportanceFactorizedWeights",
        "LaplaceKernelWeights",
        "TransformedWeights",
    }
    assert set(children) == {
        "GaussianWeights",
        "ImportanceWeights",
        "ImportanceFactorizedWeights",
        "LaplaceKernelWeights",
    }
    assert set(laws) == {
        ("gaussian-mixture", "gaussian"),
        ("random-covariance", "uniform"),
        ("random-covariance", "gaussian"),
    }
    count += 4
    workflows = []
    for profile in PROFILES:
        for family in FAMILIES:
            directory = out / f"explorer-{profile}-{family}"
            directory.mkdir()
            session = ExplorerSession(directory / "saved")
            result = session.request(
                "generate",
                {
                    "record": {
                        "recipe": {"rows": 16, "context_rows": 8},
                        "seed": 28,
                        "task_id": 10,
                    },
                    "sampling": {"family": family, "profile": profile},
                },
            )
            dump(directory / "generated.json", result)
            graph = session.request("graph", {"indices": [0, 15]})["graph"]
            dump(directory / "graph.json", graph)
            assert graph["target_max_error"] < 1e-10
            constructions = [
                n["parameters"]["weight_construction"]
                for n in graph["nodes"]
                if "weight_construction" in n.get("parameters", {})
            ]
            expected = json.loads(result["recipe_json"])["recipe"]["nodes"]
            assert len(constructions) == sum(bool(n["parents"]) for n in expected)
            assert all(v["version"] == "normalized-weights-v1" for v in constructions)
            reopened = ExplorerSession(directory / "saved").request(
                "open", {"archive": result["archive"], "task_id": 10}
            )
            dump(directory / "reopened.json", reopened)
            assert reopened["view"] == result["view"]
            replay_session = ExplorerSession(directory / "replay")
            replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
            dump(directory / "replay.json", replay)
            assert replay["recipe_json"] == result["recipe_json"]
            for index in (0, 15):
                row = session.request("row", {"index": index})
                dump(directory / f"row-{index}.json", row)
                assert replay_session.request("row", {"index": index}) == row
            count += 7
            workflows.append(
                dict(
                    profile=profile,
                    family=family,
                    status="PASS",
                    saved_archive=result["archive"],
                    trace_target_max_error=graph["target_max_error"],
                )
            )
    count += sum(admission(out, profile) for profile in PROFILES)
    source = Path(inspect.getfile(sample_recipe))
    result = dict(
        status="PASS",
        checks=count,
        recipes=len(outcomes),
        failed_draws=sum(r["status"] == "FAILED" for r in outcomes),
        workflows=workflows,
        families=dict(kinds),
        children=dict(children),
        point_laws={str(k): v for k, v in laws.items()},
        widths=sorted(widths),
        python=sys.executable,
        public_import=str(source),
        module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    dump(out / "workflow-summary.json", result)
    print(json.dumps(result))


def admission(out, profile):
    checks = 0
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

            def call(*args, **kwargs):
                value = fn(*args, **kwargs)
                if name in ("normal", "uniform", "random", "choice", "permutation"):
                    try:
                        sha = digest(value)
                    except (ValueError, TypeError):
                        sha = None
                    events.append(dict(kind=name, shape=list(np.shape(value)), sha256=sha))
                return value

            return call

    def rng(*args, **kwargs):
        return LoggedRng(original_rng(*args, **kwargs))

    def admit(self, *args, **kwargs):
        result = original_admit(self, *args, **kwargs)
        events.append(dict(kind="admission", result=result))
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
            profile=profile,
        )
    expected = recipe.limits.admit(
        recipe.nodes, rows=16, calibration_rows=0, features=recipe.task.features, outputs=1
    )
    admissions = [
        i for i, e in enumerate(events) if e["kind"] == "admission" and e["result"] == expected
    ]
    assert admissions
    allocations = []

    def matrix_arrays(record):
        data = asdict(record)
        # Every top jitter array is independently sampled; raw family arrays below are
        # additionally inspected in source for the same post-admission draw boundary.
        return [data["jitter"]]

    for node in recipe.nodes:
        if not node.parents:
            continue
        for array in matrix_arrays(node.weight_construction):
            matched = [
                i
                for i, e in enumerate(events)
                if e["kind"] == "normal" and e["sha256"] == digest(array)
            ]
            assert len(matched) == 1
            assert admissions[0] < matched[0]
            allocations.extend(matched)
            checks += 1
    assert allocations
    dump(
        out / f"admission-{profile}.json",
        dict(
            events=events,
            expected=expected,
            matching_admissions=admissions,
            allocation_events=allocations,
        ),
    )
    checks += 2
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    run(parser.parse_args().output_dir)
