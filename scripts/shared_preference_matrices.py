"""Public configurable-prior, matrix arithmetic, tamper and delivery probes."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import sys
import traceback
from dataclasses import asdict
from decimal import Decimal as D
from decimal import localcontext
from itertools import product
from pathlib import Path

import numpy as np

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def numeric_count(value):
    if isinstance(value, dict):
        return sum(
            numeric_count(v) for k, v in value.items() if k not in ("limits", "exact_scores_limit")
        )
    if isinstance(value, (tuple, list)):
        return sum(map(numeric_count, value))
    return int(type(value) in (int, float))


def vector(v):
    if isinstance(v, p.UniformWeightVector):
        return np.asarray(v.draws)
    scores = [v.log_noise_scale * z - v.decay * math.log(i + 1) for i, z in enumerate(v.draws)]
    weights = np.exp(np.asarray(scores) - max(scores))
    weights *= math.sqrt(len(weights) / sum(weights * weights))
    return weights[list(v.permutation or range(len(weights)))]


def normalize(values, jitter, scale, epsilon):
    with localcontext() as ctx:
        ctx.prec = 90
        result = []
        for row, noise in zip(values, jitter, strict=True):
            exact = [D(float(v)) + D(scale) * D(z) for v, z in zip(row, noise, strict=True)]
            den = D(epsilon) + sum(v * v for v in exact).sqrt()
            result.append([float(v / den) if den else 0.0 for v in exact])
        return np.asarray(result)


def matrix(record, trace):
    if isinstance(record, p.NormalizedWeights):
        result = normalize(
            matrix(record.source, trace), record.jitter, record.jitter_scale, record.epsilon
        )
    elif isinstance(record, p.GaussianWeights):
        result = np.asarray(record.draws)
    elif isinstance(record, p.ImportanceWeights):
        raw = np.asarray(record.gaussian) * np.asarray([vector(v) for v in record.importance])
        result = normalize(raw, np.zeros_like(raw).tolist(), 0.0, record.epsilon)
    elif isinstance(record, p.ImportanceFactorizedWeights):
        result = (np.asarray(record.left) * vector(record.diagonal)) @ np.asarray(record.right)
    elif isinstance(record, p.TransformedWeights):
        assert record.activation.kind == "identity"
        assert not record.activation.parameters
        result = matrix(record.child, trace) + record.jitter_scale * np.asarray(record.jitter)
    elif isinstance(record, p.LaplaceKernelWeights):
        cloud = record.point_construction
        points = []
        for c, z in zip(cloud.assignments, cloud.innovations, strict=True):
            values = (np.asarray(cloud.factors[c]) * vector(cloud.factor_weights[c])) @ np.asarray(
                z
            )
            scale = math.exp(cloud.scale_mean + cloud.scale_coefficient * cloud.scale_draws[c])
            points.append(np.asarray(cloud.centers[c]) + scale * values)
        points = np.asarray(points)
        stored = np.asarray((*record.left_points, *record.right_points))
        assert np.allclose(points, stored, rtol=2e-11, atol=2e-13)
        left, right = stored[: len(record.left_points)], stored[len(record.left_points) :]
        distance = np.sqrt(np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2))
        result = np.asarray(record.signs) * np.exp(-record.distance_scale * distance)
        trace.append(
            {
                "kind": "point_cloud",
                "expected": points.tolist(),
                "actual": stored.tolist(),
                "law": cloud.law,
                "base_law": cloud.base_law,
            }
        )
    else:
        raise AssertionError(type(record).__name__)
    actual = np.asarray(record.materialize())
    trace.append(
        {
            "kind": type(record).__name__,
            "actual": actual.tolist(),
            "expected": result.tolist(),
            "max_abs_error": float(np.max(np.abs(actual - result))),
        }
    )
    assert np.allclose(actual, result, rtol=3e-10, atol=2e-13)
    return result


def reject(out, key, raw):
    dump(out / (key + "-input.json"), raw)
    try:
        p.recipe_from_dict(raw)
    except (ValueError, TypeError, KeyError) as exc:
        return {"key": key, "status": "REJECTED_AS_REQUIRED", "error": str(exc)}
    raise AssertionError("accepted invalid binding: " + key)


def run(out):
    out.mkdir(parents=True)
    outcomes = []
    try:
        source = Path(inspect.getfile(p.sample_preferred_matrices))
        dump(
            out / "identity.json",
            {
                "python": sys.executable,
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        )
        controls = p.LinearRecipe(
            rows=16, context_rows=8, features=2, active_features=2, signal_features=2
        )
        base = p.sample_causal_recipe(
            p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
            task=controls,
            seed=28,
            task_id=1,
            widths=(1, 2, 3, 1),
            activations=p.CausalActivationPrior(),
        )
        dump(out / "base.json", asdict(base))
        configurations = [
            (kind, "gaussian", "gaussian-mixture", "gaussian")
            for kind in ("gaussian", "importance", "factorized")
        ]
        configurations += [
            ("kernel", "gaussian", law, innovation)
            for law, innovation in (
                ("gaussian-mixture", "gaussian"),
                ("random-covariance", "gaussian"),
                ("random-covariance", "uniform"),
            )
        ]
        configurations += [
            ("transformed", kind, "gaussian-mixture", "gaussian")
            for kind in ("gaussian", "importance", "factorized")
        ]
        configurations += [
            ("transformed", "kernel", law, innovation)
            for law, innovation in (
                ("gaussian-mixture", "gaussian"),
                ("random-covariance", "gaussian"),
                ("random-covariance", "uniform"),
            )
        ]
        last = None
        for (mode, policy), vf, config in product(
            (("meta", "shared"), ("meta", "refresh"), ("local", "shared"), ("global", "shared")),
            ("polynomial", "uniform"),
            configurations,
        ):
            family, child, law, innovation = config
            key = "-".join((mode, policy, vf, *config))
            prior = p.CausalMatrixPrior(
                mode=mode,
                categorical_policy=policy,
                vector_family=vf,
                families=(family,),
                simple_families=(child,),
                point_laws=(law,),
                covariance_bases=(innovation,),
            )
            recipe = p.sample_preferred_matrices(base, rng=np.random.default_rng(2831), prior=prior)
            raw = json.loads(json.dumps(asdict(recipe)))
            dump(out / (key + "-recipe.json"), raw)
            last = raw if family == "kernel" and vf == "polynomial" else last
            assert p.recipe_from_dict(raw) == recipe
            for field in (
                "task",
                "topology",
                "observed",
                "target",
                "activation_prior",
                "activation_probabilities",
            ):
                assert getattr(base, field) == getattr(recipe, field)
            trace = []
            for old, node in zip(base.nodes, recipe.nodes, strict=True):
                for field in (
                    "parents",
                    "bias",
                    "noise_scale",
                    "input_activation",
                    "output_activation",
                    "noise_position",
                ):
                    assert getattr(old, field) == getattr(node, field)
                if node.parents:
                    expected = matrix(node.weight_construction, trace)
                    assert np.allclose(node.weights, expected, rtol=3e-10, atol=2e-13)
                else:
                    assert old == node
            dump(out / (key + "-arithmetic.json"), trace)
            # Read paths from plain dictionaries, independently of PreferenceBinding.read.
            used = set()
            for b in raw["preference_bindings"]:
                value = raw
                for part in b["path"]:
                    value = value[part]
                if b["operation"] == "length":
                    value = len(value)
                d = raw[b["kind"] + "_preferences"]["draws"][b["draw"]]
                expected = d["values"][0] if b["kind"] == "numerical" else d["value"]
                assert value == expected
                identity = (b["kind"], b["draw"])
                assert identity not in used
                used.add(identity)
            assert len(used) == (
                len(recipe.numerical_preferences.draws) + len(recipe.categorical_preferences.draws)
            )
            extra = sum(
                numeric_count(raw[f])
                for f in ("numerical_preferences", "categorical_preferences", "preference_bindings")
            )
            assert extra == recipe.sampling_parameter_count
            generated = p.generate_task(recipe, seed=28, task_id=1)
            restored = p.generate_task(p.recipe_from_dict(raw), seed=28, task_id=1)
            assert generated.episode() == restored.episode()
            card = json.loads(generated.mechanism_json)
            admission = recipe.limits.admit(
                recipe.nodes, rows=16, calibration_rows=0, features=2, outputs=1
            )
            assert card["resource_admission"]["parameters"] == admission["parameters"] + extra
            dump(
                out / (key + "-generated.json"),
                {
                    "recipe_json": generated.recipe_json,
                    "mechanism_json": generated.mechanism_json,
                    "episode": asdict(generated.episode()),
                },
            )
            outcomes.append(
                {
                    "key": key,
                    "status": "PASS",
                    "matrix_stages": len(trace),
                    "sampling_parameters": extra,
                    "bindings": len(used),
                }
            )
            dump(out / "outcomes.json", outcomes)
        for attack in (
            "drop",
            "duplicate",
            "absent",
            "path",
            "orphan",
            "prior-relabel",
            "request-relabel",
            "value",
            "policy",
        ):
            raw = copy.deepcopy(last)
            if attack == "drop":
                raw["preference_bindings"].pop()
            elif attack == "duplicate":
                raw["preference_bindings"].append(copy.deepcopy(raw["preference_bindings"][0]))
            elif attack == "absent":
                raw["preference_bindings"][0]["draw"] = 100000
            elif attack == "path":
                raw["preference_bindings"][0]["path"][-1] = "epsilon"
            elif attack == "orphan":
                raw["categorical_preferences"]["draws"].append(
                    copy.deepcopy(raw["categorical_preferences"]["draws"][0])
                )
            elif attack == "prior-relabel":
                raw["matrix_prior"]["vector_family"] = "uniform"
            elif attack == "request-relabel":
                for collection in ("preferences", "draws"):
                    for d in raw["categorical_preferences"][collection]:
                        if d["request"]["name"] == "random_matrix_type":
                            d["request"]["choices"] = ["gaussian-weights-v1"]
                            if collection == "draws":
                                d["value"] = "gaussian-weights-v1"
            elif attack == "value":
                for b in raw["preference_bindings"]:
                    if b["kind"] == "numerical":
                        value = raw
                        for part in b["path"][:-1]:
                            value = value[part]
                        value[b["path"][-1]] *= 1.01
                        break
            else:
                raw["matrix_prior"]["mode"] = "local"
            outcomes.append(reject(out, "tamper-" + attack, raw))
        # Existing specialized profiles cannot be silently overwritten.
        for profile in ("causal-exogenous-v1", "causal-factorized-v1", "causal-matrices-v1"):
            specialized = p.sample_recipe(
                "forward", task=controls, seed=28, task_id=10, profile=profile
            )
            try:
                p.sample_preferred_matrices(specialized, rng=np.random.default_rng(28))
            except ValueError as exc:
                outcomes.append(
                    {
                        "key": "specialized-" + profile,
                        "status": "REJECTED_AS_REQUIRED",
                        "error": str(exc),
                    }
                )
            else:
                raise AssertionError("specialized construction was overwritten")
        profiles = (
            "causal-shared-matrices-v1",
            "causal-refreshed-matrices-v1",
            "causal-local-matrices-v1",
            "causal-global-matrices-v1",
        )
        for profile, family in product(profiles, ("forward", "layered", "redirection")):
            key = profile + "-" + family
            recipe = p.sample_recipe(family, task=controls, seed=28, task_id=10, profile=profile)
            generated = p.generate_task(recipe, seed=28, task_id=10)
            restored = p.generate_task(p.recipe_from_dict(asdict(recipe)), seed=28, task_id=10)
            assert generated.episode() == restored.episode()
            dump(
                out / (key + "-generated.json"),
                {
                    "recipe": asdict(recipe),
                    "episode": asdict(generated.episode()),
                    "trace": p.explain_generation(generated, [0, 15]),
                },
            )
            outcomes.append({"key": key, "status": "PASS"})
        # Four actual backend saves/reopens/replays, with row and graph inspection.
        for profile in profiles:
            folder = out / profile
            folder.mkdir()
            session = ExplorerSession(folder / "saved")
            result = session.request(
                "generate",
                {
                    "record": {
                        "recipe": {"rows": 16, "context_rows": 8},
                        "seed": 28,
                        "task_id": 10,
                    },
                    "sampling": {"family": "forward", "profile": profile},
                },
            )
            dump(folder / "result.json", result)
            graph = session.request("graph", {"indices": [0, 15]})
            dump(folder / "graph.json", graph)
            assert graph["graph"]["target_max_error"] < 1e-10
            reopened = ExplorerSession(folder / "saved").request(
                "open", {"archive": result["archive"], "task_id": 10}
            )
            dump(folder / "reopened.json", reopened)
            assert reopened["view"] == result["view"]
            replay_session = ExplorerSession(folder / "replay")
            replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
            assert replay["recipe_json"] == result["recipe_json"]
            dump(folder / "replay.json", replay)
            for index in (0, 15):
                row = session.request("row", {"index": index})
                dump(folder / f"row-{index}.json", row)
                assert replay_session.request("row", {"index": index}) == row
            outcomes.append(
                {"key": "backend-" + profile, "status": "PASS", "archive": result["archive"]}
            )
        dump(out / "outcomes.json", outcomes)
        dump(
            out / "summary.json",
            {
                "status": "PASS",
                "probes": len(outcomes),
                "configurations": 96,
                "profile_replays": 12,
                "backend_workflows": 4,
            },
        )
    except Exception as exc:
        dump(
            out / "failure.json",
            {"status": "REJECT", "error": repr(exc), "traceback": traceback.format_exc()},
        )
        dump(out / "outcomes.json", outcomes)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
