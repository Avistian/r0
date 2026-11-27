"""Independent NumPy equations and contract probes for frozen node pipelines."""

import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from custody import record_custody

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession
from r0.modules.priors.lib.causal import causal_rng, causal_values
from r0.modules.priors.lib.causal_moments import causal_moment_orders
from r0.modules.priors.lib.pipeline_ancestry import pipeline_ancestry

OUT = Path(__file__).parent
RESULTS = []


def encode(value):
    if isinstance(value, torch.Tensor | np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def save(name, value):
    (OUT / f"{name}.json").write_text(json.dumps(value, default=encode, indent=2) + "\n")


def close(name, actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(a - e))) if a.size else 0.0
    RESULTS.append(dict(name=name, max_error=error))
    np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-12, err_msg=name)


def reject(name, operation):
    try:
        operation()
    except (ValueError, TypeError, KeyError) as error:
        RESULTS.append(dict(name=name, status=type(error).__name__, error=str(error)))
        return
    raise AssertionError(f"{name}: unexpectedly accepted")


def controls(**kwargs):
    return p.LinearRecipe(
        rows=16, context_rows=8, features=3, signal_features=3, active_features=3, **kwargs
    )


def recipe_for(variant, task_type="regression"):
    l2_sample = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    warp_sample = np.array([[-1.0], [1.0]])
    save("calibration-inputs", dict(l2=l2_sample, warp=warp_sample))
    l2 = p.calibrate_l2_normalization(l2_sample, calibration_id="verifier-vector")
    warp = p.calibrate_kumaraswamy(warp_sample, a=2.0, b=1.0, calibration_id="verifier-range")
    weights = ((1.0, -0.25, 0.3), (0.5, 1.0, -0.5), (-0.75, 0.2, 1.0))
    construction = p.FactorizedWeights(
        left=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        right=weights,
        diagonal=(1.0, 1.0, 1.0),
    )
    nodes = (
        p.ActivatedCausalNode(
            bias=(0.5, -0.75, 1.2), noise_scale=(0.2, 0.3, 0.4), noise_position="after_output"
        ),
        p.MatrixCausalNode(
            parents=(0,),
            weights=weights,
            bias=(0.1, -0.2, 0.3),
            noise_scale=(0.2, 0.3, 0.1),
            weight_construction=construction,
            output_activation=p.CausalActivation("square"),
            noise_response=p.NoiseResponse(offset=0.4, strength=0.7, power=2.0),
            noise_position="after_output",
        ),
    )
    pipelines = (
        p.CausalNodePipeline(
            channels=3,
            stages=(p.CausalActivation("square"), p.CausalActivation("tanh")),
            channel_importance=(2.0, -0.5, 0.0),
            l2=l2,
            converters=(
                p.NumericalConverter(0, warp, variant),
                p.NumericalConverter(1, variant="historical-tuple"),
            ),
            node_importance=3.0,
        ),
        p.CausalNodePipeline(
            channels=3,
            stages=(p.CausalActivation("tanh"),),
            channel_importance=(0.8, -0.4, 2.0),
            converters=(
                p.NumericalConverter(0, warp, variant),
                p.NumericalConverter(1, warp, "readout-only"),
            ),
            node_importance=1.7,
        ),
    )
    return p.CausalPipelineRecipe(
        task=controls(
            task_type=task_type, classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type]
        ),
        nodes=nodes,
        pipelines=pipelines,
        observed=((0, 0), (0, 1), (0, 2)),
        target=tuple((1, j) for j in range(3 if task_type == "multiclass" else 1)),
        version="causal-v2",
    )


def branch(value, variant, importance, innovation, second_identity=False):
    state, readout = value.copy(), value.copy()
    warp = np.clip((value + 1.0) / (2.0 + 1e-30), 0.0, 1.0) ** 2
    if variant != "readout-only":
        state[:, 0] = warp[:, 0]
    if variant != "historical-tuple":
        readout[:, 0] = warp[:, 0]
    if not second_identity:
        readout[:, 1] = warp[:, 1]
    state = importance * (state + innovation)
    readout[:, 2] = state[:, 2]
    return state, readout


def numerical(variant):
    recipe = recipe_for(variant)
    raw, readouts, scales, steps = [], [], [], []
    states, innovations = causal_values(
        recipe.nodes,
        16,
        lambda *keys: causal_rng(125, 9, *keys),
        "cpu",
        pipelines=recipe.pipelines,
        exogenous_values=raw,
        readout_values=readouts,
        noise_scales=scales,
        pipeline_steps=steps,
    )
    f = 1.0 / (math.sqrt(12.5) + 1e-8)
    root_pre = np.tile(
        np.tanh(np.array([0.5, -0.75, 1.2]) ** 2) * np.array([2.0, -0.5, 0.0]) * f, (16, 1)
    )
    n0 = np.asarray(raw[0]) * [0.2, 0.3, 0.4]
    s0, r0 = branch(root_pre, variant, 3.0, n0, second_identity=True)
    affine = s0 @ np.array(recipe.nodes[1].weights) + [0.1, -0.2, 0.3]
    scale1 = np.array([0.2, 0.3, 0.1]) * (0.4 + 0.7 * np.abs(affine) ** 2)
    n1 = np.asarray(raw[1]) * scale1
    pre1 = np.tanh(affine**2) * [0.8, -0.4, 2.0]
    s1, r1 = branch(pre1, variant, 1.7, n1)
    expected = dict(state=[s0, s1], readout=[r0, r1], innovation=[n0, n1])
    save(
        f"numerical-{variant}",
        dict(
            recipe=asdict(recipe),
            raw=raw,
            expected=expected,
            states=states,
            readouts=readouts,
            scales=scales,
            steps=steps,
            l2_factor=f,
        ),
    )
    for i in range(2):
        close(f"{variant}-state-{i}", states[i], expected["state"][i])
        close(f"{variant}-readout-{i}", readouts[i], expected["readout"][i])
        close(f"{variant}-innovation-{i}", innovations[i], expected["innovation"][i])
    close(f"{variant}-pre-affine-noise-response", scales[1], scale1)
    task = p.generate_task(recipe, seed=125, task_id=9)
    close(f"{variant}-public-input", task.inputs, r0)
    close(f"{variant}-public-target", task.scores, r1[:, :1])
    trace = p.explain_generation(task, [15, 0, 7])
    save(f"trace-{variant}", trace)
    by_id = {n["id"]: n for n in trace["nodes"]}
    for i in range(2):
        close(
            f"{variant}-trace-state-{i}",
            by_id[f"causal-{i}"]["values"],
            expected["state"][i][[15, 0, 7]],
        )
        close(
            f"{variant}-trace-readout-{i}",
            by_id[f"readout-{i}"]["values"],
            expected["readout"][i][[15, 0, 7]],
        )
    assert by_id["inputs"]["parents"] == ["readout-0"]
    assert by_id["signal"]["parents"] == ["readout-1"]
    assert by_id["pipeline-1-input"]["parents"] == ["causal-0"]
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    altered = replace(recipe, task=replace(recipe.task, rows=31, context_rows=11))
    extended = p.generate_task(altered, seed=125, task_id=9)
    close(f"{variant}-row-count-independence", extended.inputs[:16], task.inputs)
    close(f"{variant}-context-independence", extended.scores[:16], task.scores)
    assert asdict(altered)["pipelines"] == asdict(recipe)["pipelines"]
    replay_reads = []
    causal_values(
        recipe.nodes,
        16,
        lambda *keys: causal_rng(125, 9, *keys),
        "cpu",
        [15, 0, 7],
        pipelines=recipe.pipelines,
        readout_values=replay_reads,
    )
    close(f"{variant}-chunk-independence", replay_reads[1], r1[[15, 0, 7]])
    return recipe


def moments():
    heavy = p.ExogenousDistribution(inputs=p.InputDistribution(distribution="student_t", df=0.5))
    root = p.StochasticCausalNode(exogenous=heavy, noise_position="after_output")
    child = p.StochasticCausalNode(
        parents=(0,), weights=((1.0,),), exogenous=heavy, noise_position="after_output"
    )
    warp = p.calibrate_kumaraswamy(
        np.array([[-1.0], [1.0]]), a=1.0, b=1.0, calibration_id="verifier-heavy-bounding"
    )
    l2 = p.calibrate_l2_normalization(
        np.array([[-1.0], [1.0]]), calibration_id="finite-not-population-proof"
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    base = p.CausalPipelineRecipe(
        task=task,
        nodes=(root, child),
        pipelines=(p.CausalNodePipeline(1), p.CausalNodePipeline(1)),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    for variant in ("historical-tuple", "historical-graph", "readout-only"):
        candidate = replace(
            base,
            pipelines=(
                base.pipelines[0],
                p.CausalNodePipeline(1, converters=(p.NumericalConverter(0, warp, variant),)),
            ),
        )
        readouts = []
        state = causal_moment_orders(
            candidate.nodes, pipelines=candidate.pipelines, readout_orders=readouts
        )
        assert state == [[0.5], [0.5]]
        assert readouts == [[0.5], [0.5 if variant == "historical-tuple" else math.inf]]
        if variant == "historical-tuple":
            reject(
                "heavy-historical-target-normalization",
                lambda candidate=candidate: replace(candidate, normalize=True),
            )
        else:
            normalized = replace(candidate, normalize=True, calibration_rows=32)
            generated = p.generate_task(normalized, seed=125, task_id=8)
            raw, cal = [], []
            causal_values(
                normalized.nodes,
                16,
                lambda *k: causal_rng(125, 8, *k),
                "cpu",
                pipelines=normalized.pipelines,
                readout_values=raw,
            )
            causal_values(
                normalized.nodes,
                32,
                lambda *k: causal_rng(125, 8, "calibration", *k),
                "cpu",
                pipelines=normalized.pipelines,
                readout_values=cal,
            )
            target, calibration = np.asarray(raw[1]), np.asarray(cal[1])
            expected = (target - calibration.mean(axis=0)) / calibration.std(axis=0)
            close(f"{variant}-captured-target-normalization", generated.scores, expected)
            save(
                f"moments-{variant}",
                dict(
                    recipe=asdict(normalized),
                    target=target,
                    calibration=calibration,
                    expected=expected,
                    actual=generated.scores,
                    state_orders=state,
                    readout_orders=readouts,
                ),
            )
    reject(
        "l2-before-bound",
        lambda: replace(
            base, pipelines=(base.pipelines[0], p.CausalNodePipeline(1, stages=(l2, warp)))
        ),
    )
    accepted = replace(
        base, pipelines=(base.pipelines[0], p.CausalNodePipeline(1, stages=(warp,), l2=l2))
    )
    assert causal_moment_orders(accepted.nodes, pipelines=accepted.pipelines) == [[0.5], [0.5]]
    standard = p.calibrate_source_activation(
        p.CausalActivation("square"),
        np.array([[-1.0], [1.0], [2.0]]),
        rng=np.random.default_rng(25),
        scale=1.0,
        calibration_id="verifier-standard-square",
    )
    root3 = replace(
        root,
        exogenous=p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="student_t", df=3.0)
        ),
    )
    reject(
        "standard-output-needs-second-moment",
        lambda: replace(
            base,
            nodes=(root3, child),
            pipelines=(base.pipelines[0], p.CausalNodePipeline(1, stages=(standard,))),
        ),
    )
    reject(
        "standard-input-needs-second-moment",
        lambda: replace(
            base, pipelines=(base.pipelines[0], p.CausalNodePipeline(1, stages=(standard,)))
        ),
    )
    zeroed = replace(
        base,
        pipelines=(base.pipelines[0], p.CausalNodePipeline(1, channel_importance=(0.0,), l2=l2)),
    )
    p.generate_task(zeroed, seed=125, task_id=2)
    RESULTS.append(dict(name="moment-placement", status="PASS"))


def ancestry():
    nodes = (
        p.ActivatedCausalNode(
            bias=(0.0, 0.0), noise_scale=(1.0, 1.0), noise_position="after_output"
        ),
        p.ActivatedCausalNode(
            parents=(0,),
            weights=((1.0, 0.0), (0.0, 1.0)),
            bias=(0.0, 0.0),
            noise_scale=(1.0, 1.0),
            noise_position="after_output",
        ),
        p.ActivatedCausalNode(
            parents=(1,),
            weights=((1.0,), (1.0,)),
            noise_scale=(0.0,),
            noise_position="after_output",
        ),
    )
    pipelines = (
        p.CausalNodePipeline(2),
        p.CausalNodePipeline(
            2,
            stages=(p.CausalActivation("softmax"),),
            channel_importance=(1.0, 0.0),
            converters=(p.NumericalConverter(0), p.NumericalConverter(1)),
        ),
        p.CausalNodePipeline(1),
    )
    actual = pipeline_ancestry(nodes, pipelines)
    expected = {
        (0, 0): {(0, 0)},
        (0, 1): {(0, 1)},
        (1, 0): {(0, 0), (0, 1)},
        (1, 1): set(),
        (2, 0): {(0, 0), (0, 1), (1, 0), (1, 1)},
    }
    save(
        "ancestry",
        dict(
            actual={str(k): v for k, v in actual.items()},
            expected={str(k): v for k, v in expected.items()},
        ),
    )
    assert actual == expected
    RESULTS.append(dict(name="ancestry-mixing-zero-later-noise", status="PASS"))


def contracts(recipe):
    # Independent accounting: 3+3 node widths, fan-in 3, 3 features, 1 target.
    # Activation parameters=0; matrix extras: exogenous 1 + response 3 + factors 21.
    card = json.loads(p.generate_task(recipe, seed=125, task_id=9).mechanism_json)
    save("resource-observed", card["resource_admission"])
    admitted = card["resource_admission"]
    exact = replace(
        recipe,
        limits=p.CausalLimits(
            parameters=admitted["parameters"], working_bytes=admitted["estimated_working_bytes"]
        ),
    )
    p.generate_task(exact, seed=125, task_id=9)
    for field, value in (
        ("parameters", admitted["parameters"] - 1),
        ("working_bytes", admitted["estimated_working_bytes"] - 1),
    ):
        reject(
            f"minus-one-{field}",
            lambda field=field, value=value: replace(
                recipe, limits=replace(exact.limits, **{field: value})
            ),
        )
    record = json.loads(json.dumps(asdict(recipe)))
    for field, value in (
        ("channels", True),
        ("channels", 2),
        ("version", "bad"),
        ("node_importance", 0),
        ("node_importance", float("nan")),
        ("channel_importance", [1.0]),
        ("unexpected", 1),
    ):
        bad = deepcopy(record)
        bad["pipelines"][0][field] = value
        reject(f"malformed-{field}-{value}", lambda bad=bad: p.recipe_from_dict(bad))
    for field, value in (
        ("channel", -1),
        ("channel", 3),
        ("channel", True),
        ("variant", "bad"),
        ("transform", {"kind": "square"}),
    ):
        bad = deepcopy(record)
        bad["pipelines"][0]["converters"][0][field] = value
        reject(f"converter-{field}-{value}", lambda bad=bad: p.recipe_from_dict(bad))
    bad = deepcopy(record)
    bad["pipelines"][0]["converters"].append(bad["pipelines"][0]["converters"][0])
    reject("duplicate-converter", lambda: p.recipe_from_dict(bad))
    bad = deepcopy(record)
    bad["nodes"][0]["noise_position"] = "before_output"
    reject("wrong-noise-position", lambda: p.recipe_from_dict(bad))


def workflows():
    for variant in ("historical-tuple", "historical-graph", "readout-only"):
        for task_type in ("regression", "binary", "multiclass"):
            label = f"workflow-{variant}-{task_type}"
            session = ExplorerSession(OUT / label)
            generated = session.request(
                "generate",
                {
                    "record": {
                        "recipe": asdict(recipe_for(variant, task_type)),
                        "seed": 125,
                        "task_id": 9,
                    }
                },
            )
            graph = session.request("graph", {"indices": [0, 7, 15]})
            reopened = ExplorerSession(OUT / label).request(
                "open", {"archive": generated["archive"], "task_id": 9}
            )
            assert reopened["view"] == generated["view"]
            replay = ExplorerSession(OUT / (label + "-replay"))
            replayed = replay.request("generate", {"record_json": generated["recipe_json"]})
            assert replayed["recipe_json"] == generated["recipe_json"]
            assert replay.request("row", {"index": 7}) == session.request("row", {"index": 7})
            save(
                label,
                dict(
                    generated=generated,
                    graph=graph,
                    reopened=reopened,
                    replayed=replayed,
                    row=session.request("row", {"index": 7}),
                ),
            )
            RESULTS.append(dict(name=label, status="PASS"))


def main():
    record_custody(OUT / "verify-start.custody.json")
    try:
        for variant in ("historical-tuple", "historical-graph", "readout-only"):
            recipe = numerical(variant)
        moments()
        ancestry()
        contracts(recipe)
        workflows()
    finally:
        save("results", RESULTS)
        record_custody(OUT / "verify-end.custody.json")
    print(json.dumps(dict(status="PASS", checks=len(RESULTS))))


if __name__ == "__main__":
    main()
