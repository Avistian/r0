from dataclasses import asdict

import numpy as np
from common import (
    OUT,
    close,
    draw,
    fitted,
    ok,
    save,
    task_settings,
    trace,
)

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def verify_pipelines(idx):
    act, actfun = fitted(
        "square", np.array([[-2.7], [-1.8], [-0.4], [0.1], [0.9], [2.2], [3.6]]), 487
    )
    root = p.RootInputCausalNode(
        root_scale=(-0.8, 1.1, 0.5),
        bias=(0.3, -0.7, 0.2),
        noise_scale=(0.1, 0.3, 0.2),
        noise_response=p.NoiseResponse(offset=0.4, strength=0.2, power=2),
    )
    weights = np.array([[1.2, 0.3, -0.4], [-0.2, 0.8, 0.1], [0.5, -0.6, 1.1]])
    child = p.StochasticCausalNode(
        parents=(0,),
        weights=tuple(tuple(float(v) for v in row) for row in weights),
        bias=(0.1, -0.2, 0.3),
        noise_scale=(0.05, 0.06, 0.09),
        noise_position="after_output",
    )
    l2 = p.calibrate_l2_normalization(
        np.array([[1.0, -0.3, 0.2], [0.5, 0.8, -0.7], [-0.9, 0.4, 1.1]]),
        calibration_id="own-l2-sample",
    )
    factor = 1 / (
        np.sqrt(
            np.mean(
                np.sum(
                    np.array([[1.0, -0.3, 0.2], [0.5, 0.8, -0.7], [-0.9, 0.4, 1.1]]) ** 2, axis=1
                )
            )
        )
        + 1e-8
    )
    for variant in ["historical-tuple", "historical-graph", "readout-only"]:
        pipe = p.CausalNodePipeline(
            channels=3,
            stages=(p.CausalActivation("tanh"),),
            channel_importance=(1.3, -0.7, 0.5),
            l2=l2,
            converters=(p.NumericalConverter(channel=0, transform=act, variant=variant),),
            node_importance=1.8,
        )
        for kind in ["regression", "binary", "multiclass"]:
            name = variant + "-" + kind
            seed = 732
            tid = 12
            rec = p.CausalPipelineRecipe(
                task=task_settings(kind),
                nodes=(root, child),
                observed=((0, 0), (0, 1)),
                target=tuple((1, j) for j in range(3 if kind == "multiclass" else 1)),
                version="causal-v2",
                pipelines=(pipe, p.CausalNodePipeline(channels=3, node_importance=0.9)),
            )
            t = p.generate_task(rec, seed=seed, task_id=tid)
            tr, v = trace(t, idx)
            rd = draw(root.root_exogenous, 48, 3, seed, tid, root=True)
            aff = np.array(root.bias) + rd * np.array(root.root_scale)
            pre = np.tanh(aff) * np.array(pipe.channel_importance) * factor
            conv = actfun(pre[:, 0:1])
            statepre = pre.copy()
            if variant != "readout-only":
                statepre[:, 0:1] = conv
            inn = (
                np.array(root.noise_scale)
                * (0.4 + 0.2 * abs(aff) ** 2)
                * draw(root.exogenous, 48, 3, seed, tid)
            )
            state = (statepre + inn) * 1.8
            readout = state.copy()
            readout[:, 0:1] = pre[:, 0:1] if variant == "historical-tuple" else conv
            childvalue = (
                state @ weights
                + np.array(child.bias)
                + draw(child.exogenous, 48, 3, seed, tid, node=1) * np.array(child.noise_scale)
            ) * 0.9
            save(
                name + ".json",
                dict(
                    recipe=asdict(rec),
                    trace=tr,
                    expected=dict(
                        root=rd,
                        affine=aff,
                        pipeline_pre=pre,
                        converted=conv,
                        state_pre=statepre,
                        innovation=inn,
                        state=state,
                        readout=readout,
                        child=childvalue,
                    ),
                    actual=dict(inputs=t.inputs, scores=t.scores, targets=t.targets),
                ),
            )
            close(name + "-pipeline-l2", v["pipeline-0-l2"], pre[idx])
            close(name + "-state", v["causal-0"], state[idx])
            close(name + "-input", t.inputs, readout[:, :2])
            close(name + "-scores", t.scores, childvalue[:, : len(rec.target)])
            directory = OUT / "explorer" / name
            session = ExplorerSession(directory)
            result = session.request(
                "generate", {"record": {"recipe": asdict(rec), "seed": seed, "task_id": tid}}
            )
            graph = session.request("graph", {"indices": idx})["graph"]
            reopen = ExplorerSession(directory)
            opened = reopen.request("open", {"archive": result["archive"], "task_id": tid})
            ok(name + "-saved-view", opened["view"] == result["view"])
            rep = ExplorerSession(directory / "replay")
            repeated = rep.request("generate", {"record_json": result["recipe_json"]})
            ok(name + "-replayed-recipe", repeated["recipe_json"] == result["recipe_json"])
            close(
                name + "-saved-graph-root",
                next(n["values"] for n in graph["nodes"] if n["id"] == "root-input-0"),
                rd[idx],
            )
            ok(
                name + "-saved-last-row",
                session.request("row", {"index": 47}) == rep.request("row", {"index": 47}),
            )
            save(
                name + "-workflow.json",
                dict(
                    generation=result,
                    opened=opened,
                    graph=graph,
                    row=session.request("row", {"index": 47}),
                ),
            )
    return act, root, pipe, child
