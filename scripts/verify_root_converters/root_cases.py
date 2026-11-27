import json
import subprocess
from dataclasses import asdict, replace

import numpy as np
from common import (
    HEAD,
    REPO,
    close,
    custody,
    draw,
    ok,
    plain,
    save,
    trace,
)

from r0.modules import priors as p


def verify_roots():
    custody("custody-before.json")
    ok(
        "exact-head",
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == HEAD,
    )
    laws = {
        "gaussian": p.ExogenousDistribution(inputs=p.InputDistribution(location=0.3, scale=1.2)),
        "uniform": p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="uniform", location=-0.7, scale=0.6)
        ),
        "student": p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="student_t", df=3.7)
        ),
        "mixture": p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="mixture", mixture_gap=1.3)
        ),
        "truncated": p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="student_t", df=0.8, bound=1.1)
        ),
        "ball": p.ExogenousDistribution(geometry="ball"),
        "factor": p.ExogenousDistribution(
            factor=((1.0, -0.5, 0.2), (0.3, 0.0, 0.9), (0.0, 0.7, -0.4))
        ),
    }
    idx = [47, 0, 23, 6, 19]
    for k, law in laws.items():
        for pos in ["before_output", "after_output"]:
            name = k + "-" + pos
            seed = 917
            tid = 38
            root = p.RootInputCausalNode(
                root_exogenous=law,
                root_scale=(-1.2, 0.0, 0.8),
                bias=(0.4, -0.8, 0.2),
                noise_scale=(0.3, 0.5, 0.2),
                noise_position=pos,
                output_activation=p.CausalActivation("tanh"),
                exogenous=p.ExogenousDistribution(
                    inputs=p.InputDistribution(
                        distribution="mixture", location=0.1, scale=0.8, mixture_gap=0.4
                    )
                ),
                noise_response=p.NoiseResponse(offset=0.2, strength=0.3, power=1.5),
            )
            recipe = plain(root)
            task = p.generate_task(recipe, seed=seed, task_id=tid)
            tr, vs = trace(task, idx)
            rd = draw(law, 48, 3, seed, tid, root=True)
            ex = draw(root.exogenous, 48, 3, seed, tid)
            aff = np.asarray(root.bias) + rd * np.asarray(root.root_scale)
            sc = np.asarray(root.noise_scale) * (0.2 + 0.3 * abs(aff) ** 1.5)
            inn = sc * ex
            val = np.tanh(aff + inn) if pos == "before_output" else np.tanh(aff) + inn
            save(
                name + ".json",
                dict(
                    recipe=asdict(recipe),
                    trace=tr,
                    root_draw=rd,
                    innovation_draw=ex,
                    affine=aff,
                    noise_scale=sc,
                    innovation=inn,
                    expected_value=val,
                    actual_inputs=task.inputs,
                    actual_scores=task.scores,
                ),
            )
            for label, expected in [
                ("root-input-0", rd),
                ("exogenous-0", ex),
                ("noise-scale-0", sc),
                ("innovation-0", inn),
                ("causal-0", val),
            ]:
                close(name + "-" + label, vs[label], expected[idx])
            close(name + "-inputs", task.inputs, val[:, :2])
            close(name + "-scores", task.scores, val[:, 2:])
            ok(name + "-streams-differ", not np.array_equal(rd, ex))
            long = p.generate_task(
                replace(recipe, task=replace(recipe.task, rows=71, context_rows=31)),
                seed=seed,
                task_id=tid,
            )
            close(name + "-row-prefix-inputs", long.inputs[:48], task.inputs, 0)
            close(name + "-row-prefix-scores", long.scores[:48], task.scores, 0)
            quiet = p.generate_task(
                replace(recipe, nodes=(replace(root, noise_scale=(0.0, 0.0, 0.0)),)),
                seed=seed,
                task_id=tid,
            )
            _, qv = trace(quiet, idx)
            close(name + "-noise-edit-root-stream", qv["root-input-0"], rd[idx])
            restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
            ok(name + "-roundtrip", restored == recipe)
    return idx
