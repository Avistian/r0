"""Public recipes checked with independent NumPy equation reconstruction."""

import json
from dataclasses import asdict, replace

import numpy as np
from common import check, close, custody, rejects, save

from r0.modules import priors as p

sample2 = np.array([[-3.0, 2.0], [1.0, 8.0], [5.0, 5.0]])
sample1 = np.array([[-2.0], [4.0], [7.0]])
std2 = p.calibrate_standardization(sample2, calibration_id="recipe-two")
std1 = p.calibrate_standardization(sample1, calibration_id="recipe-one")
mean2, inv2 = sample2.mean(0), 1 / (sample2.std(0) + 1e-4)
mean1, inv1 = sample1.mean(0), 1 / (sample1.std(0) + 1e-4)
save("calibration.json", dict(sample1=sample1, sample2=sample2, standard1=std1, standard2=std2))
task_controls = p.LinearRecipe(
    rows=18, context_rows=9, features=2, signal_features=2, active_features=2
)
for placement in ["input", "output"]:
    for position in ["before_output", "after_output"]:
        label = placement + "-" + position
        node = p.ActivatedCausalNode(
            parents=(0,),
            weights=((1.25,), (-0.75,)),
            bias=(0.6,),
            noise_scale=(0.4,),
            noise_position=position,
            **{placement + "_activation": std2 if placement == "input" else std1},
        )
        recipe = p.CausalRecipe(
            task=task_controls,
            nodes=(p.CausalNode(bias=(0.2, -0.7), noise_scale=(1.2, 0.8)), node),
            observed=((0, 0), (0, 1)),
            target=((1, 0),),
            version="causal-v2",
        )
        restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
        check(label + "-decode", restored, recipe)
        task = p.generate_task(restored, seed=601, task_id=63)
        graph = p.explain_generation(task, [17, 0, 9])
        save(label + "-trace.json", graph)
        steps = {n["id"]: n for n in graph["nodes"]}
        x = task.inputs.numpy()[[17, 0, 9]]
        noise = np.array(steps["innovation-1"]["values"])
        if placement == "input":
            x = (x - mean2) * inv2
        expected = x @ np.array([[1.25], [-0.75]]) + 0.6
        if position == "before_output":
            expected = expected + noise
        if placement == "output":
            expected = (expected - mean1) * inv1
        if position == "after_output":
            expected = expected + noise
        close(label + "-equation", task.scores.numpy()[[17, 0, 9]], expected)
        card = json.loads(task.mechanism_json)
        check(label + "-moments", card["moment_contract"]["target_orders"], ["all"])
        check(label + "-ancestry", card["active"], [0, 1])
        changed = replace(recipe, task=replace(task_controls, rows=24, context_rows=7))
        longer = p.generate_task(changed, seed=601, task_id=63)
        close(label + "-row-extension", longer.scores.numpy()[:18], task.scores.numpy())
        # Geometry from explicit equations, not product resource helper outputs.
        params = 4 + 4 + (5 if placement == "input" else 3)
        temp = 20 if placement == "input" else 10
        working = 8 * 18 * (6 * 3 + 2 * 2 + 2 * 2 + 4 + 4 + temp) + 128 * params
        exact = replace(recipe, limits=p.CausalLimits(parameters=params, working_bytes=working))
        admitted = json.loads(p.generate_task(exact, seed=601, task_id=63).mechanism_json)[
            "resource_admission"
        ]
        check(label + "-admission-parameters", admitted["parameters"], params)
        check(label + "-admission-bytes", admitted["estimated_working_bytes"], working)
        for field, value in [("parameters", params - 1), ("working_bytes", working - 1)]:
            rejects(
                label + "-below-" + field,
                lambda field=field, value=value, recipe=recipe, exact=exact: replace(
                    recipe, limits=replace(exact.limits, **{field: value})
                ),
            )

# Full source ordering, with three independent state/readout converter contracts.
cal = np.array([[-0.8, -0.3], [0.2, 0.4], [0.7, 0.9]])
standard = p.calibrate_standardization(cal, calibration_id="pipeline-standard")
weighted_cal = (cal - cal.mean(0)) / (cal.std(0) + 1e-4) * np.array([1.7, -0.6])
l2 = p.calibrate_l2_normalization(weighted_cal, calibration_id="pipeline-l2")
l2_factor = 1 / (np.sqrt(np.mean(np.sum(weighted_cal**2, axis=1))) + 1e-8)
warp_cal = np.array([[-2.0], [2.0]])
warp = p.calibrate_kumaraswamy(warp_cal, a=2.0, b=3.0, calibration_id="pipeline-warp")
save(
    "pipeline-calibration.json",
    dict(
        sample=cal,
        post_weights=weighted_cal,
        warp_sample=warp_cal,
        standard=standard,
        l2=l2,
        expected_l2=l2_factor,
        warp=warp,
    ),
)
for variant in ["readout-only", "historical-tuple", "historical-graph"]:
    root = p.RootInputCausalNode(
        bias=(0.4, -0.2),
        root_scale=(0.9, 1.3),
        noise_scale=(0.1, 0.2),
        output_activation=p.CausalActivation("tanh"),
    )
    child = p.ActivatedCausalNode(
        parents=(0,),
        weights=((1.2,), (-0.7,)),
        bias=(0.4,),
        noise_scale=(0.15,),
        noise_position="after_output",
    )
    pipeline = p.CausalNodePipeline(
        channels=2,
        stages=(standard,),
        channel_importance=(1.7, -0.6),
        l2=l2,
        converters=(p.NumericalConverter(channel=0, transform=warp, variant=variant),),
        node_importance=2.3,
    )
    recipe = p.CausalPipelineRecipe(
        task=task_controls,
        nodes=(root, child),
        pipelines=(pipeline, p.CausalNodePipeline(channels=1)),
        observed=((0, 0), (0, 1)),
        target=((1, 0),),
        version="causal-v2",
    )
    task = p.generate_task(recipe, seed=647, task_id=34)
    graph = p.explain_generation(task, [17, 0, 9])
    save(variant + "-trace.json", graph)
    steps = {n["id"]: n for n in graph["nodes"]}
    raw = np.array(steps["root-input-0"]["values"])
    innovation = np.array(steps["innovation-0"]["values"])
    activated = np.tanh(raw * [0.9, 1.3] + [0.4, -0.2])
    standardized = (activated - cal.mean(0)) / (cal.std(0) + 1e-4)
    close(variant + "-stage-values", steps["pipeline-0-stage-0"]["values"], standardized)
    scaled = standardized * [1.7, -0.6] * l2_factor
    z = np.clip((scaled[:, 0] + 2) / 4, 0, 1)
    warped = 1 - (1 - z**2) ** 3
    propagated = scaled.copy()
    if variant != "readout-only":
        propagated[:, 0] = warped
    state = (propagated + innovation) * 2.3
    observed = state.copy()
    observed[:, 0] = scaled[:, 0] if variant == "historical-tuple" else warped
    expected = state @ np.array([[1.2], [-0.7]]) + 0.4 + np.array(steps["innovation-1"]["values"])
    close(variant + "-features", task.inputs.numpy()[[17, 0, 9]], observed)
    close(variant + "-descendant", task.scores.numpy()[[17, 0, 9]], expected)
    check(variant + "-ancestry", json.loads(task.mechanism_json)["active"], [0, 1])
    record = steps["pipeline-0-stage-0"]["parameters"]["pipeline"]["stages"][0]
    check(
        variant + "-trace-record",
        json.loads(json.dumps(record)),
        json.loads(json.dumps(asdict(standard))),
    )
    check(
        variant + "-roundtrip", p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), recipe
    )
custody()
