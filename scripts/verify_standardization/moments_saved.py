"""Independent moment admission cases and saved three-task workflows."""

import json
from dataclasses import asdict, replace

import numpy as np
from common import OUT, check, close, custody, rejects, save

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession

standard = p.calibrate_standardization(
    np.array([[-2.0], [3.0], [8.0]]), calibration_id="moment-fit"
)
controls = p.LinearRecipe(rows=20, context_rows=8, features=1, signal_features=1, active_features=1)
heavy = p.ExogenousDistribution(inputs=p.InputDistribution(distribution="student_t", df=1.5))
finite = p.ExogenousDistribution(inputs=p.InputDistribution(distribution="student_t", df=3.0))
root = p.RootInputCausalNode(noise_scale=(0.0,), root_exogenous=heavy)
child = p.ActivatedCausalNode(
    parents=(0,), weights=((1.3,),), noise_scale=(0.0,), noise_position="after_output"
)


def equation(nodes, normalize=False):
    return p.CausalRecipe(
        task=controls,
        nodes=nodes,
        observed=((0, 0),),
        target=((len(nodes) - 1, 0),),
        normalize=normalize,
        calibration_rows=32,
        version="causal-v2",
    )


def pipeline(stages, zero=False):
    return p.CausalPipelineRecipe(
        task=controls,
        nodes=(root, child),
        pipelines=(
            p.CausalNodePipeline(
                channels=1, stages=stages, channel_importance=(0.0,) if zero else ()
            ),
            p.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )


for placement in ["input", "output"]:
    standardized = replace(child, **{placement + "_activation": standard})
    rejects("heavy-" + placement, lambda standardized=standardized: equation((root, standardized)))
    good = equation((replace(root, root_exogenous=finite), standardized))
    task = p.generate_task(good, seed=51, task_id=82)
    check(
        "finite-" + placement,
        json.loads(task.mechanism_json)["moment_contract"]["target_orders"],
        [3.0],
    )
    save("finite-" + placement + ".json", dict(recipe=good, card=json.loads(task.mechanism_json)))
    good = equation((replace(root, output_activation=p.CausalActivation("tanh")), standardized))
    check(
        "bounded-" + placement,
        json.loads(p.generate_task(good, seed=51, task_id=82).mechanism_json)["moment_contract"][
            "target_orders"
        ],
        ["all"],
    )
rejects("later-zero-does-not-rescue", lambda: pipeline((standard,), zero=True))
rejects("later-bounded-does-not-rescue", lambda: pipeline((standard, p.CausalActivation("tanh"))))
check(
    "bounded-before-stage",
    isinstance(pipeline((p.CausalActivation("tanh"), standard)), p.CausalPipelineRecipe),
)
zero = p.CausalActivation("identity", (("output_scale", 0.0),))
check("zero-before-stage", isinstance(pipeline((zero, standard)), p.CausalPipelineRecipe))
# Square of t_3 lacks second moments; bounded conversion first restores them.
finite_root = replace(root, root_exogenous=finite, output_activation=p.CausalActivation("square"))
rejects(
    "finite-input-square-infinite-output",
    lambda: equation((finite_root, replace(child, output_activation=standard))),
)
unknown_root = replace(
    root, root_exogenous=p.ExogenousDistribution(), output_activation=p.CausalActivation("exp")
)
rejects(
    "unknown-exp-moments",
    lambda: equation((unknown_root, replace(child, output_activation=standard))),
)
# Output placement precedes heavy innovation in after_output, follows it otherwise.
for position in ["before_output", "after_output"]:
    noisy = p.StochasticCausalNode(
        parents=(0,),
        weights=((1.3,),),
        noise_scale=(0.1,),
        noise_position=position,
        exogenous=heavy,
        output_activation=standard,
    )
    nodes = (replace(root, root_exogenous=p.ExogenousDistribution()), noisy)
    if position == "before_output":
        rejects("heavy-noise-before-standard", lambda nodes=nodes: equation(nodes))
    else:
        candidate = equation(nodes)
        check("heavy-noise-after-standard", isinstance(candidate, p.CausalRecipe))
        check(
            "heavy-noise-moment-card",
            json.loads(p.generate_task(candidate, seed=51, task_id=82).mechanism_json)[
                "moment_contract"
            ]["target_orders"],
            [1.5],
        )
        rejects("heavy-target-normalization", lambda nodes=nodes: equation(nodes, normalize=True))
# Global equation gates catch prior fitted stages even if bounded before final Standardize.
l2 = p.calibrate_l2_normalization(np.array([[-1.0], [1.0]]), calibration_id="moment-l2")
wrapped = p.calibrate_source_activation(
    p.CausalActivation("tanh"),
    np.array([[-1.0], [1.0]]),
    rng=np.random.default_rng(97),
    scale=2.0,
    calibration_id="moment-wrapper",
)
for label, fitted in [("l2", l2), ("wrapped", wrapped)]:
    middle = replace(child, output_activation=fitted)
    bounded = replace(child, parents=(1,), output_activation=p.CausalActivation("tanh"))
    last = replace(child, parents=(2,), output_activation=standard)
    rejects(
        "upstream-fitted-" + label,
        lambda middle=middle, bounded=bounded, last=last: equation((root, middle, bounded, last)),
    )
    legacy = replace(last, output_activation=p.CausalActivation())
    check(
        "legacy-admission-" + label,
        isinstance(equation((root, middle, bounded, legacy)), p.CausalRecipe),
    )
# Wrong record widths/versions through public recipe decoding.
valid = equation((replace(root, root_exogenous=finite), replace(child, output_activation=standard)))
for label, field, value in [
    ("version", "calibration_version", "unsupported"),
    ("kind", "kind", "identity"),
    ("width", "inverse_std", [1.0, 2.0]),
    ("parameters", "parameters", [["a", 2.0]]),
]:
    record = json.loads(json.dumps(asdict(valid)))
    record["nodes"][1]["output_activation"][field] = value
    rejects("decode-" + label, lambda record=record: p.recipe_from_dict(record))

sample = np.array([[-3.0, 2.0, 5.0], [1.0, 4.0, 5.0], [7.0, 9.0, 5.0]])
stage = p.calibrate_standardization(sample, calibration_id="saved-three-channel-fit")
save("saved-calibration.json", dict(sample=sample, fitted=stage))
for task_type in ["regression", "binary", "multiclass"]:
    controls = p.LinearRecipe(
        rows=20,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
    )
    node = p.ActivatedCausalNode(
        parents=(0,),
        weights=((1.3, -0.9, 0.4),),
        bias=(0.2, -0.1, 0.7),
        noise_scale=(0.2, 0.3, 0.1),
        noise_position="after_output",
    )
    recipe = p.CausalPipelineRecipe(
        task=controls,
        nodes=(p.RootInputCausalNode(noise_scale=(0.0,)), node),
        pipelines=(
            p.CausalNodePipeline(channels=1),
            p.CausalNodePipeline(channels=3, stages=(stage,)),
        ),
        observed=((0, 0),),
        target=tuple((1, j) for j in range(3 if task_type == "multiclass" else 1)),
        normalize=True,
        calibration_rows=32,
        version="causal-v2",
    )
    original = ExplorerSession(OUT / task_type)
    created = original.request(
        "generate", {"record": {"recipe": asdict(recipe), "seed": 503, "task_id": 89}}
    )
    graph = original.request("graph", {"indices": [19, 0, 8]})["graph"]
    save(task_type + "-generated.json", dict(created=created, graph=graph))
    check(task_type + "-trace-target", graph["target_max_error"] < 1e-12)
    steps = {n["id"]: n for n in graph["nodes"]}
    stage_values = np.array(steps["pipeline-1-stage-0"]["values"])
    # Inspect true parent input values, independently reconstruct affine and frozen standardization.
    parent = np.array(steps["root-input-0"]["values"])
    affine = parent @ np.array([[1.3, -0.9, 0.4]]) + [0.2, -0.1, 0.7]
    expected = (affine - sample.mean(0)) / (sample.std(0) + 1e-4)
    close(task_type + "-saved-equation", stage_values, expected)
    frozen = steps["pipeline-1-stage-0"]["parameters"]["pipeline"]["stages"][0]
    check(
        task_type + "-frozen-record",
        json.loads(json.dumps(frozen)),
        json.loads(json.dumps(asdict(stage))),
    )
    reopened = ExplorerSession(OUT / task_type).request(
        "open", {"archive": created["archive"], "task_id": 89}
    )
    check(task_type + "-reopen", reopened["view"], created["view"])
    replay = ExplorerSession(OUT / (task_type + "-replay"))
    repeated = replay.request("generate", {"record_json": created["recipe_json"]})
    check(task_type + "-replay-record", repeated["recipe_json"], created["recipe_json"])
    check(
        task_type + "-replay-row",
        replay.request("row", {"index": 19}),
        original.request("row", {"index": 19}),
    )
    save(task_type + "-reopened-replay.json", dict(reopened=reopened, replay=repeated))
custody()
