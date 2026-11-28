"""Independent public-entry-point checks; exact oracle uses binary Fractions."""

import copy
import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import torch

from r0.modules.priors import (
    CausalLimits,
    CausalNode,
    CausalRecipe,
    FactorizedWeights,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)


def authored(left, diagonal, right, weights, **extra):
    incoming, outgoing = len(left), len(right[0])
    data = asdict(
        CausalRecipe(
            task=LinearRecipe(
                rows=16,
                context_rows=8,
                features=incoming,
                signal_features=incoming,
                active_features=incoming,
            ),
            nodes=(
                CausalNode(bias=(0.0,) * incoming, noise_scale=(1.0,) * incoming),
                CausalNode(
                    parents=(0,),
                    weights=tuple(tuple(r) for r in weights),
                    bias=(0.0,) * outgoing,
                    noise_scale=(0.0,) * outgoing,
                ),
            ),
            observed=tuple((0, j) for j in range(incoming)),
            target=((1, 0),),
            version="causal-v2",
        )
    )
    data["nodes"][1]["weight_construction"] = {
        "version": "factorized-weights-v1",
        "left": left,
        "diagonal": diagonal,
        "right": right,
        **extra,
    }
    return data


def rational_product(left, diagonal, right):
    return [
        [
            sum(
                (
                    Fraction(left[i][k]) * Fraction(diagonal[k]) * Fraction(right[k][j])
                    for k in range(len(diagonal))
                ),
                Fraction(),
            )
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def rounded(left, diagonal, right):
    return [[float(x) for x in row] for row in rational_product(left, diagonal, right)]


def reject(data, fragment):
    try:
        recipe_from_dict(data)
    except (ValueError, TypeError) as error:
        if fragment not in str(error):
            raise AssertionError(str(error)) from error
        return str(error)
    raise AssertionError("unexpectedly admitted " + fragment)


def numerical_checks(mark, out):
    huge, tiny, sub = float(2**1023), 2.0**-1023, 2.0**-1074
    fixtures = [
        ("original-overflow", [[huge]], [4.0], [[tiny]]),
        ("intermediate-underflow", [[tiny]], [tiny], [[huge]]),
        ("opposite-order-overflow", [[tiny]], [huge], [[2.0]]),
        ("overflow-cancellation", [[huge, huge, 1.0]], [4.0, 4.0, 1.0], [[1.0], [-1.0], [1.0]]),
        ("underflow-then-recovered", [[sub]], [0.5], [[2.0]]),
        ("subnormal-exact", [[sub]], [1.0], [[1.0]]),
        ("subnormal-round-up", [[sub]], [1.5], [[1.0]]),
        ("half-subnormal-round-zero", [[sub]], [0.5], [[1.0]]),
        ("zero-coefficient", [[huge]], [0.0], [[huge]]),
        ("zero-right", [[huge]], [4.0], [[0.0]]),
        ("zero-left", [[0.0]], [huge], [[huge]]),
        ("negative-subnormal", [[-sub]], [1.0], [[1.0]]),
        ("signed-cancellation", [[huge, -huge]], [4.0, 4.0], [[1.0], [1.0]]),
        (
            "rectangular-extreme",
            [[huge, tiny], [-huge, tiny]],
            [4.0, 0.5],
            [[tiny, tiny, 0.0], [huge, -huge, huge]],
        ),
    ]
    retained = []
    for name, left, diagonal, right in fixtures:
        dense = rounded(left, diagonal, right)
        data = authored(left, diagonal, right, dense)
        recipe = recipe_from_dict(data)
        got = [list(row) for row in recipe.nodes[1].weights]
        assert got == dense, (name, got, dense)
        assert [list(row) for row in recipe.nodes[1].weight_construction.materialize()] == dense
        replay = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
        assert replay == recipe
        retained.append({"name": name, "input": data, "oracle": dense})
        mark("Fraction exact construction " + name)
    (out / "rational_extremes.json").write_text(json.dumps(retained, indent=2) + "\n")
    for left in (math.nextafter(huge, 0), huge, math.nextafter(huge, math.inf)):
        for coefficient in (math.nextafter(2.0, 0), 2.0, math.nextafter(2.0, math.inf)):
            data = authored(
                [[left]], [coefficient], [[tiny]], rounded([[left]], [coefficient], [[tiny]])
            )
            recipe_from_dict(data)
    mark("nine neighboring finite overflow-boundary products")
    data = authored([[huge]], [4.0], [[1.0]], [[1.0]])
    reject(data, "nonfinite factorized projection")
    mark("genuinely nonrepresentable final projection rejects")
    left, diagonal, right = [[huge, tiny], [huge, tiny]], [4.0, 0.5], [[tiny, tiny], [huge, huge]]
    data = authored(left, diagonal, right, rounded(left, diagonal, right), exact_terms_limit=8)
    recipe_from_dict(data)
    data["nodes"][1]["weight_construction"]["exact_terms_limit"] = 7
    reject(data, "exact arithmetic term limit")
    mark("exact arithmetic term boundary 2*2*2 admits eight rejects seven")
    for invalid in (0, -1, True, 1.5, 2**63):
        data["nodes"][1]["weight_construction"]["exact_terms_limit"] = invalid
        reject(data, "positive term limit")
    mark("exact arithmetic invalid budgets rejected")
    # Numeric rank deficiency is valid, without requiring orthogonal factors.
    left, diagonal, right = [[1, 1], [2, 2]], [1, 0], [[3, 4], [5, 6]]
    recipe_from_dict(authored(left, diagonal, right, [[3, 4], [6, 8]]))
    mark("nonzero rank-deficient projection admitted")


def authored_checks(mark, out):
    data = authored([[1, 2], [3, 4]], [2, 0.5], [[5, 6], [7, 8]], [[17, 20], [44, 52]])
    assert rounded([[1, 2], [3, 4]], [2, 0.5], [[5, 6], [7, 8]]) == [[17, 20], [44, 52]]
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=11)
    torch.testing.assert_close(
        task.scores[:, 0], 17 * task.inputs[:, 0] + 44 * task.inputs[:, 1], rtol=1e-14, atol=1e-14
    )
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 29
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=11
        ).episode()
        == task.episode()
    )
    trace = explain_generation(task, [0, 15])
    node = next(n for n in trace["nodes"] if n["id"] == "causal-1")
    assert node["parameters"]["weight_construction"]["diagonal"] == (2, 0.5)
    assert trace["target_max_error"] < 1e-10
    mark("worked math/29 parameters/selected trace/JSON replay")
    for field, value, fragment in (
        ("diagonal", [-1, 0.5], "nonnegative"),
        ("diagonal", [1], "matching rank"),
        ("left", [[1, 2], [3]], "rectangular"),
        ("right", [[1, 2]], "dimensions"),
        ("left", [[math.inf, 2], [3, 4]], "finite"),
        ("right", [[5], [7]], "shape"),
        ("version", "fake", "version"),
    ):
        invalid = copy.deepcopy(data)
        invalid["nodes"][1]["weight_construction"][field] = value
        reject(invalid, fragment)
    mark("invalid dimensions/version/nonfinite/negative factors rejected")
    for part in ("dense", "factors"):
        invalid = copy.deepcopy(data)
        if part == "dense":
            invalid["nodes"][1]["weights"] = [[18, 20], [44, 52]]
        else:
            invalid["nodes"][1]["weight_construction"]["left"][0][0] = 2
        reject(invalid, "disagree")
    mark("dense and factor tampering rejected")
    invalid = copy.deepcopy(data)
    invalid["limits"]["parameters"] = 28
    with (
        patch.object(
            FactorizedWeights,
            "materialize",
            side_effect=AssertionError("materialize before admission"),
        ),
        patch.object(torch, "tensor", side_effect=AssertionError("tensor before admission")),
        patch.object(np, "asarray", side_effect=AssertionError("array before admission")),
    ):
        reject(invalid, "parameter limit")
    mark("authored parameter admission precedes factor materialization and tensor allocation")
    data["limits"]["parameters"] = 29
    recipe_from_dict(data)
    mark("authored exact resource boundary admits 29 rejects 28")


def admission_arithmetic(recipe):
    widths = [len(n.bias) for n in recipe.nodes]
    dense = sum(len(n.bias) * (2 + sum(widths[p] for p in n.parents)) for n in recipe.nodes)
    extra = 0
    for node in recipe.nodes:
        extra += len(node.input_activation.parameters) + len(node.output_activation.parameters)
        if node.parents:
            c = node.weight_construction
            extra += 7 + sum(map(len, c.left)) + sum(map(len, c.right)) + len(c.diagonal)
    total = dense + extra
    fan_in = max(sum(widths[p] for p in n.parents) for n in recipe.nodes)
    workextra = max((8 * len(n.bias) for n in recipe.nodes if n.parents), default=0)
    outputs = recipe.task.classes if recipe.task.task_type == "multiclass" else 1
    peakrows = max(recipe.task.rows, recipe.calibration_rows if recipe.normalize else 0)
    working = (
        8
        * peakrows
        * (6 * sum(widths) + 2 * fan_in + 2 * recipe.task.features + 4 * outputs + 4 + workextra)
        + 128 * total
    )
    return {
        "nodes": len(widths),
        "largest_node_channels": max(widths),
        "total_channels": sum(widths),
        "parameters": total,
        "estimated_working_bytes": working,
    }


def sampled_checks(mark, out):
    controls = LinearRecipe(rows=16, context_rows=8)
    summary, samples, products, rank_hist = [], [], [], {}
    for family in ("forward", "layered", "redirection"):
        for seed in range(24):
            recipe = sample_recipe(
                family, task=controls, seed=seed, task_id=10, profile="causal-factorized-v1"
            )
            assert all(len(n.bias) in (1, 2, 4) for n in recipe.nodes)
            assert len(recipe.activation_prior.choices) == 26
            assert recipe.construction == "activation-factorized-v1"
            nodes = []
            for node in recipe.nodes:
                if not node.parents:
                    continue
                c = node.weight_construction
                n, m, k = len(node.weights), len(node.bias), len(c.diagonal)
                assert 1 <= k <= min(n, m)
                assert c.shape == (n, m)
                assert c.diagonal == (1 / math.sqrt(n * k),) * k
                expected = np.asarray(c.left) @ np.asarray(c.right) / math.sqrt(n * k)
                np.testing.assert_allclose(node.weights, expected, rtol=1e-13, atol=1e-14)
                assert np.linalg.matrix_rank(np.asarray(node.weights), tol=1e-10) <= k
                samples.extend(v for row in c.left + c.right for v in row)
                products.extend(v * math.sqrt(n) for row in node.weights for v in row)
                key = f"{min(n, m)}:{k}"
                rank_hist[key] = rank_hist.get(key, 0) + 1
                nodes.append({"n": n, "m": m, "k": k})
            entry = {
                "family": family,
                "seed": seed,
                "nodes": nodes,
                "admission": admission_arithmetic(recipe),
            }
            try:
                task = generate_task(recipe, seed=seed, task_id=10)
                assert json.loads(task.mechanism_json)["resource_admission"] == entry["admission"]
                entry["generation"] = "PASS"
            except ValueError as error:
                entry["generation"] = "FAILED_VALIDATION"
                entry["error"] = str(error)
            summary.append(entry)
    # Population variance of an entry is k*(1/(n*k))*E[U^2]*E[V^2] = 1/n.
    # Empirical bounds are broad; RNG instrumentation below checks the law itself.
    assert abs(float(np.mean(samples))) < 0.15
    assert 0.8 < float(np.mean(np.square(samples))) < 1.2
    assert 0.65 < float(np.mean(np.square(products))) < 1.35
    (out / "sampled_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mark(
        "72 sampled recipes factor shapes/rank/scaling and independent admission arithmetic",
        factors=len(samples),
        entry_scaled_second_moment=float(np.mean(np.square(products))),
        factor_second_moment=float(np.mean(np.square(samples))),
        rank_histogram=rank_hist,
        failed_generation=sum(x["generation"] != "PASS" for x in summary),
    )
    recipe = sample_recipe(
        "redirection", task=controls, seed=28, task_id=10, profile="causal-factorized-v1"
    )
    data = asdict(recipe)
    child = next(n for n in data["nodes"] if n["parents"])
    child.pop("weight_construction")
    reject(data, "factorized sampling requires")
    mark("missing sampled factor provenance rejected")
    task = generate_task(recipe, seed=28, task_id=10)
    long = generate_task(replace(recipe, task=replace(controls, rows=32)), seed=28, task_id=10)
    for field in ("inputs", "targets", "scores", "probabilities"):
        a, b = getattr(task, field), getattr(long, field)
        if a is not None:
            assert torch.equal(a, b[:16]), field
    assert task.source_task_id != long.source_task_id
    repartitioned = generate_task(
        replace(recipe, task=replace(controls, context_rows=6)), seed=28, task_id=10
    )
    assert task.source_task_id == repartitioned.source_task_id
    assert task.episode_id != repartitioned.episode_id
    replay = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert generate_task(replay, seed=28, task_id=10).episode() == task.episode()
    trace = explain_generation(task, [0, 15])
    assert trace["target_max_error"] < 1e-10
    for index, node in enumerate(recipe.nodes):
        if node.parents:
            found = next(n for n in trace["nodes"] if n["id"] == f"causal-{index}")
            assert found["parameters"]["weight_construction"] == asdict(node.weight_construction)
    mark("sampled prefix/identity/replay and all child factor traces")


def rng_and_admission_checks(mark, out):
    events, generators = [], []
    original_rng, original_admit = np.random.default_rng, CausalLimits.admit_dimensions

    class Proxy:
        def __init__(self, gen, ident):
            self.gen, self.ident = gen, ident

        def __getattr__(self, name):
            fn = getattr(self.gen, name)
            if name not in ("normal", "integers"):
                return fn

            def wrapped(*args, **kwargs):
                value = fn(*args, **kwargs)
                events.append(
                    {
                        "kind": name,
                        "id": self.ident,
                        "args": args,
                        "kwargs": kwargs,
                        "value": np.asarray(value).tolist(),
                    }
                )
                return value

            return wrapped

    def rng(*args, **kwargs):
        proxy = Proxy(original_rng(*args, **kwargs), len(generators))
        generators.append(proxy)
        return proxy

    def admit(self, *args, **kwargs):
        events.append({"kind": "admit", "extra": kwargs.get("extra_parameters", 0)})
        return original_admit(self, *args, **kwargs)

    with (
        patch.object(np.random, "default_rng", rng),
        patch.object(CausalLimits, "admit_dimensions", admit),
    ):
        recipe = sample_recipe(
            "redirection",
            task=LinearRecipe(rows=16, context_rows=8),
            seed=28,
            task_id=10,
            profile="causal-factorized-v1",
        )
    # The profile's single explicit RNG is first; other generators are keyed mechanism streams.
    draws = [(i, e) for i, e in enumerate(events) if e.get("id") == 0]
    normals = [(i, e) for i, e in draws if e["kind"] == "normal"]
    children = [n for n in recipe.nodes if n.parents]
    assert len(normals) == 2 * len(children)
    rankdraws = [(i, e) for i, e in draws if e["kind"] == "integers"][-len(children) :]
    assert len(rankdraws) == len(children)
    admission = next(
        i for i, e in enumerate(events) if e["kind"] == "admit" and i > rankdraws[-1][0]
    )
    assert admission < normals[0][0]
    for j, node in enumerate(children):
        draw = rankdraws[j][1]
        assert draw["args"] == (1, min(len(node.weights), len(node.bias)) + 1)
        assert draw["value"] == len(node.weight_construction.diagonal)
        for e, wanted in (
            (normals[2 * j][1], node.weight_construction.left),
            (normals[2 * j + 1][1], node.weight_construction.right),
        ):
            assert e["args"] == ()
            assert tuple(e["kwargs"]) == ("size",)
            assert e["value"] == [list(r) for r in wanted]
    (out / "rng_admission_events.json").write_text(json.dumps(events, indent=2) + "\n")
    mark(
        "uniform rank RNG/Gaussian standard factors and admission before factor arrays",
        admitted_event=admission,
        first_factor_array_event=normals[0][0],
    )
    # An actual rejection at that gate must prevent every factor-array draw.
    events.clear()
    generators.clear()
    gate_extra = admission_arithmetic(recipe)["parameters"] - sum(
        len(n.bias) * (2 + len(n.weights)) for n in recipe.nodes
    )

    def fail_at_factor_gate(self, *args, **kwargs):
        if kwargs.get("extra_parameters") == gate_extra:
            raise ValueError("independent factor allocation guard")
        return original_admit(self, *args, **kwargs)

    with (
        patch.object(np.random, "default_rng", rng),
        patch.object(CausalLimits, "admit_dimensions", fail_at_factor_gate),
    ):
        try:
            sample_recipe(
                "redirection",
                task=LinearRecipe(rows=16, context_rows=8),
                seed=28,
                task_id=10,
                profile="causal-factorized-v1",
            )
        except ValueError as error:
            if str(error) != "independent factor allocation guard":
                raise AssertionError(str(error)) from error
        else:
            raise AssertionError("factor admission gate did not reject")
    assert not any(e.get("id") == 0 and e["kind"] == "normal" for e in events)
    mark("factor admission rejection prevents all factor-array allocation")


def explorer_checks(mark, out):
    from r0.modules.explorer import ExplorerSession

    for family in ("forward", "layered", "redirection"):
        directory = out / ("explorer_" + family)
        session = ExplorerSession(directory)
        result = session.request(
            "generate",
            {
                "record": {"recipe": {"rows": 16, "context_rows": 8}, "seed": 28, "task_id": 10},
                "sampling": {"family": family, "profile": "causal-factorized-v1"},
            },
        )
        record = json.loads(result["recipe_json"])
        assert record["recipe"]["construction"] == "activation-factorized-v1"
        graph = session.request("graph", {"indices": [0, 15]})["graph"]
        assert graph["target_max_error"] < 1e-10
        reopened = ExplorerSession(directory).request(
            "open", {"archive": result["archive"], "task_id": 10}
        )
        assert reopened["view"] == result["view"]
        replay_session = ExplorerSession(out / ("replay_" + family))
        replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
        assert replay["recipe_json"] == result["recipe_json"]
        assert replay_session.request("row", {"index": 15}) == session.request("row", {"index": 15})
        (out / ("explorer_" + family + "_record.json")).write_text(result["recipe_json"] + "\n")
        mark(f"{family} explorer backend generate/trace/save/reopen/replay row")
